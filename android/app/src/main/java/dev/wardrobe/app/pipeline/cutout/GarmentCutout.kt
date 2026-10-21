package dev.wardrobe.app.pipeline.cutout

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import dev.wardrobe.app.domain.cutout.CutoutReport
import dev.wardrobe.app.domain.cutout.CutoutSettings
import dev.wardrobe.app.domain.cutout.SegClass
import dev.wardrobe.app.domain.cutout.alignToBody
import dev.wardrobe.app.domain.cutout.changedMask
import dev.wardrobe.app.domain.cutout.matchLighting
import dev.wardrobe.app.pipeline.pose.PoseDetector
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.io.File
import java.nio.ByteOrder

/** One photo after the models ran: upright RGB (work size), pose landmarks (px), person mask, segmenter classes. */
class Analysis(
    val rgb: Mat,              // CV_8UC3
    val landmarks: FloatArray, // 33 x (x px, y px, visibility)
    val person: Mat,           // CV_32F 0..1
    val classes: Mat,          // CV_8U segmenter labels
) {
    val width get() = rgb.cols()
    val height get() = rgb.rows()
    fun release() { rgb.release(); person.release(); classes.release() }
}

class CutoutResult(val rgba: Bitmap, val report: CutoutReport)

/**
 * Garment cut-out on the tablet, ported from prototype/pipeline.py: align the garment photo to the body
 * figure with pose landmarks, then keep the person pixels that differ from the body figure (minus
 * skin/hair), clean up, and feather the edge. Output is in the body figure's frame, so it can be drawn
 * straight onto the body figure. Not thread-safe; close when done.
 */
class GarmentCutout(context: Context, private val s: CutoutSettings = CutoutSettings()) : Closeable {

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV failed to load" }
    }

    private val pose = PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(PoseDetector.MODEL_ASSET).build())
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(1)
            .setOutputSegmentationMasks(true)
            .build(),
    )

    private val segmenter = ImageSegmenter.createFromOptions(
        context,
        ImageSegmenter.ImageSegmenterOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(SEGMENTER_ASSET).build())
            .setRunningMode(RunningMode.IMAGE)
            .setOutputCategoryMask(true)
            .setOutputConfidenceMasks(false)
            .build(),
    )

    /** Loads a JPEG upright (EXIF) at work size and runs the pose model and the segmenter on it. */
    fun analyse(photo: File): Analysis {
        val bitmap = loadUpright(photo, s.workLongSide)
        val image = BitmapImageBuilder(bitmap).build()
        val w = bitmap.width
        val h = bitmap.height

        val poseResult = pose.detect(image)
        val lm = poseResult.landmarks().firstOrNull()
            ?: throw IllegalStateException("no person found in ${photo.name}")
        val landmarks = FloatArray(33 * 3)
        lm.forEachIndexed { i, p ->
            landmarks[i * 3] = p.x() * w
            landmarks[i * 3 + 1] = p.y() * h
            landmarks[i * 3 + 2] = p.visibility().orElse(0f)
        }
        val maskImage = poseResult.segmentationMasks().get().first()
        val person = Mat(maskImage.height, maskImage.width, CvType.CV_32F).apply {
            val floats = ByteBufferExtractor.extract(maskImage).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val arr = FloatArray(floats.remaining()).also { floats.get(it) }
            put(0, 0, arr)
        }.resizedTo(w, h, Imgproc.INTER_LINEAR)

        val categoryImage = segmenter.segment(image).categoryMask().get()
        val classes = Mat(categoryImage.height, categoryImage.width, CvType.CV_8U).apply {
            val bytes = ByteBufferExtractor.extract(categoryImage)
            val arr = ByteArray(bytes.remaining()).also { bytes.get(it) }
            put(0, 0, arr)
        }.resizedTo(w, h, Imgproc.INTER_NEAREST)

        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        bitmap.recycle()
        return Analysis(rgb, landmarks, person, classes)
    }

    /** Cuts the garment out of [garment] using [body] (the body figure, same view) as the "before" picture. */
    fun cutOut(garment: Analysis, body: Analysis): CutoutResult {
        val warnings = mutableListOf<String>()
        val w = body.width
        val h = body.height
        val n = w * h

        // 1. Align garment photo onto the body figure (similarity transform from matching landmarks).
        val (m, info) = alignToBody(garment.landmarks, body.landmarks, maxOf(w, h), s, warnings)
        val affine = Mat(2, 3, CvType.CV_64F).apply { put(0, 0, *m.asRowMajor()) }
        val size = Size(w.toDouble(), h.toDouble())
        fun warp(src: Mat, interp: Int) = Mat().also { Imgproc.warpAffine(src, it, affine, size, interp) }
        val rgbG = warp(garment.rgb, Imgproc.INTER_LINEAR)
        val personG = warp(garment.person, Imgproc.INTER_LINEAR)
        val classesG = warp(garment.classes, Imgproc.INTER_NEAREST)
        val ones = Mat.ones(garment.height, garment.width, CvType.CV_8U)
        val validMat = warp(ones, Imgproc.INTER_NEAREST)
        ones.release()

        val personGArr = personG.floats()
        val personBArr = body.person.floats()
        val classesGArr = classesG.bytes()
        val classesBArr = body.classes.bytes()
        val valid = validMat.bytes()
        val isPersonG = BooleanArray(n) { personGArr[it] > s.personThreshold && valid[it].toInt() != 0 }
        val isPersonB = BooleanArray(n) { personBArr[it] > s.personThreshold }

        // 2. Colours in LAB, lightly blurred; match lighting on the shared background.
        val labB = toLab(body.rgb)
        val labG = toLab(rgbG)
        val background = BooleanArray(n) { !isPersonG[it] && !isPersonB[it] && valid[it].toInt() != 0 }
        val shift = matchLighting(labG, labB, background, warnings)

        // 3. Nearest reliable "before" pixel: any body-figure person pixel except hair/face.
        val notReliable = Mat(h, w, CvType.CV_8U)
        val nr = ByteArray(n) { i ->
            val c = classesBArr[i].toInt()
            if (isPersonB[i] && c != SegClass.HAIR && c != SegClass.FACE_SKIN) 0 else 1
        }
        notReliable.put(0, 0, nr)
        val distMat = Mat()
        val labelsMat = Mat()
        Imgproc.distanceTransformWithLabels(notReliable, distMat, labelsMat, Imgproc.DIST_L2, 5, Imgproc.DIST_LABEL_PIXEL)
        val dist = distMat.floats()
        val labels = IntArray(n).also { labelsMat.get(0, 0, it) }
        // DIST_LABEL_PIXEL gives each zero (reliable) pixel its own label; map labels back to pixel indices.
        val labelToIndex = IntArray(n + 1) { -1 }
        for (i in 0 until n) if (nr[i].toInt() == 0) labelToIndex[labels[i]] = i
        val nearest = IntArray(n) { labelToIndex.getOrElse(labels[it]) { -1 } }
        listOf(notReliable, distMat, labelsMat).forEach { it.release() }

        val changed = changedMask(labG, labB, nearest, dist, s)

        // 4. Garment = person, not skin/hair, changed vs body figure. Then clean up.
        val raw = Mat(h, w, CvType.CV_8U)
        raw.put(0, 0, ByteArray(n) { i ->
            val c = classesGArr[i].toInt()
            val skinHair = c == SegClass.HAIR || c == SegClass.BODY_SKIN || c == SegClass.FACE_SKIN
            if (isPersonG[i] && !skinHair && changed[i]) 1 else 0
        })
        val keep = cleanup(raw)
        raw.release()

        // 5. Feathered alpha, limited by the person mask.
        val keepF = Mat()
        keep.convertTo(keepF, CvType.CV_32F)
        Imgproc.GaussianBlur(keepF, keepF, Size(0.0, 0.0), s.featherPx)
        val soft = keepF.floats()
        var kept = 0
        val keepArr = keep.bytes()
        val alpha = ByteArray(n) { i ->
            if (keepArr[i].toInt() != 0) kept++
            val a = soft[i] * (personGArr[i] * 1.5f).coerceIn(0f, 1f)
            (a * 255f).toInt().coerceIn(0, 255).toByte()
        }
        val area = kept.toFloat() / n
        if (kept == 0) warnings += "nothing kept: garment looks identical to the body figure"

        val bitmap = toRgbaBitmap(rgbG, alpha)
        listOf(affine, rgbG, personG, classesG, validMat, keep, keepF).forEach { it.release() }
        return CutoutResult(
            bitmap,
            CutoutReport(info.first, m.scale, m.rotationDeg, info.second, shift[0], area, warnings),
        )
    }

    private fun cleanup(mask: Mat): Mat {
        val m = Mat()
        Imgproc.morphologyEx(mask, m, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(9.0, 9.0)))
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(m, labels, stats, centroids, 8)
        if (count > 1) {
            val areas = IntArray(count) { stats.get(it, Imgproc.CC_STAT_AREA)[0].toInt() }
            val largest = (1 until count).maxOf { areas[it] }
            val keepLabel = BooleanArray(count) { it > 0 && areas[it] >= s.minComponentFrac * largest }
            val lab = IntArray(labels.rows() * labels.cols()).also { labels.get(0, 0, it) }
            m.put(0, 0, ByteArray(lab.size) { if (keepLabel[lab[it]]) 1 else 0 })
        }
        listOf(labels, stats, centroids).forEach { it.release() }
        return m
    }

    override fun close() {
        pose.close()
        segmenter.close()
    }

    companion object {
        const val SEGMENTER_ASSET = "models/selfie_multiclass_256x256.tflite"

        /** Decodes with EXIF rotation applied, downscaled so the long side is at most [longSide]. */
        fun loadUpright(file: File, longSide: Int): Bitmap {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(file.path, bounds)
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= longSide) sample *= 2
            val decoded = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
                ?: throw IllegalStateException("cannot read ${file.name}")
            val degrees = when (ExifInterface(file.path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            val scale = minOf(1f, longSide.toFloat() / maxOf(decoded.width, decoded.height))
            val matrix = Matrix().apply { postRotate(degrees); postScale(scale, scale) }
            val upright = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
            if (upright !== decoded) decoded.recycle()
            return upright.copy(Bitmap.Config.ARGB_8888, false).also { if (it !== upright) upright.recycle() }
        }

        /** Real LAB (L 0..100), after a 5x5 blur so JPEG noise and 1 px misalignment don't count as change. */
        private fun toLab(rgb: Mat): FloatArray {
            val blurred = Mat()
            Imgproc.GaussianBlur(rgb, blurred, Size(5.0, 5.0), 0.0)
            val f = Mat()
            blurred.convertTo(f, CvType.CV_32FC3, 1.0 / 255)
            val lab = Mat()
            Imgproc.cvtColor(f, lab, Imgproc.COLOR_RGB2Lab)
            val out = FloatArray(rgb.rows() * rgb.cols() * 3).also { lab.get(0, 0, it) }
            listOf(blurred, f, lab).forEach { it.release() }
            return out
        }

        private fun toRgbaBitmap(rgb: Mat, alpha: ByteArray): Bitmap {
            val w = rgb.cols()
            val h = rgb.rows()
            val px = ByteArray(w * h * 3).also { rgb.get(0, 0, it) }
            val colors = IntArray(w * h) { i ->
                ((alpha[i].toInt() and 0xFF) shl 24) or ((px[i * 3].toInt() and 0xFF) shl 16) or
                    ((px[i * 3 + 1].toInt() and 0xFF) shl 8) or (px[i * 3 + 2].toInt() and 0xFF)
            }
            return Bitmap.createBitmap(colors, w, h, Bitmap.Config.ARGB_8888)
        }

        private fun Mat.floats() = FloatArray(rows() * cols()).also { get(0, 0, it) }
        private fun Mat.bytes() = ByteArray(rows() * cols()).also { get(0, 0, it) }
        private fun Mat.resizedTo(w: Int, h: Int, interp: Int): Mat {
            if (cols() == w && rows() == h) return this
            val out = Mat()
            Imgproc.resize(this, out, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, interp)
            release()
            return out
        }
    }
}
