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
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import dev.wardrobe.app.domain.cutout.BaseOutfit
import dev.wardrobe.app.domain.cutout.CutoutSettings
import dev.wardrobe.app.domain.cutout.DroppedPiece
import dev.wardrobe.app.domain.cutout.PieceSettings
import dev.wardrobe.app.domain.cutout.PieceType
import dev.wardrobe.app.domain.cutout.alignPointsFor
import dev.wardrobe.app.domain.cutout.alignToBody
import dev.wardrobe.app.domain.cutout.extractPieces
import dev.wardrobe.app.pipeline.pose.PoseDetector
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteOrder

/** One photo after the models ran, at work size. */
class PhotoAnalysis(
    val rgb: Mat,               // CV_8UC3, upright
    val landmarks: FloatArray,  // 33 x (x px, y px, visibility)
    val person: BooleanArray,   // pose model's person mask, thresholded
    val lab: FloatArray,        // CIELAB per pixel (lightly blurred), interleaved
    val labels: ByteArray,      // clothes model label per pixel
) {
    val width get() = rgb.cols()
    val height get() = rgb.rows()
    fun release() = rgb.release()
}

/** One garment piece found in one view. */
class PieceResult(
    val type: PieceType,
    /** Cleaned piece mask in the garment photo's frame (used for tags). */
    val mask: BooleanArray,
    /** The model's region for this piece before base-colour removal (used for sheerness). */
    val region: BooleanArray,
    val baseFrac: Float,
    /** Piece mask moved into the body figure's frame (used for side-view thickness). */
    val maskInBody: BooleanArray,
    /** Transparent cut-out in the body figure's frame, ready to draw on the body figure. */
    val rgbaInBody: Bitmap,
    val alignPoints: Int,
    val alignErrorPx: Double,
    val areaFrac: Float,
    val warnings: List<String>,
)

/**
 * Garment cut-out (method C, mirrors prototype/clothes.py): the clothes model splits the photo into pieces,
 * the base outfit is removed by colour, each piece is cleaned up and aligned onto the body figure.
 * Not thread-safe; close when done.
 */
class GarmentCutout(
    context: Context,
    private val s: CutoutSettings = CutoutSettings(),
    private val pieceSettings: PieceSettings = PieceSettings(),
) : Closeable {

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
    private val segmenter = ClothesSegmenter(context)

    /**
     * Loads a JPEG upright at work size and runs the pose and clothes models on it. With [cache], the model
     * results (landmarks, person mask, labels) are read from / written to that file, so the body figure's
     * photos are only analysed once.
     */
    fun analyse(photo: File, cache: File? = null): PhotoAnalysis {
        if (cache != null && cache.exists()) {
            readCached(photo, cache)?.let { return it }
        }
        return analyseFresh(photo).also { a -> cache?.let { writeCache(a, it) } }
    }

    private fun readCached(photo: File, cache: File): PhotoAnalysis? = try {
        DataInputStream(cache.inputStream().buffered()).use { inp ->
            val w = inp.readInt()
            val h = inp.readInt()
            val landmarks = FloatArray(33 * 3) { inp.readFloat() }
            val labels = ByteArray(w * h).also { inp.readFully(it) }
            val packed = ByteArray((w * h + 7) / 8).also { inp.readFully(it) }
            val person = BooleanArray(w * h) { (packed[it / 8].toInt() shr (it % 8)) and 1 == 1 }
            val bitmap = loadUpright(photo, s.workLongSide)
            if (bitmap.width != w || bitmap.height != h) return null
            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)
            val rgb = Mat()
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            rgba.release()
            bitmap.recycle()
            PhotoAnalysis(rgb, landmarks, person, toLab(rgb), labels)
        }
    } catch (e: java.io.IOException) {
        null
    }

    private fun writeCache(a: PhotoAnalysis, cache: File) {
        cache.parentFile?.mkdirs()
        val tmp = File(cache.path + ".tmp")
        DataOutputStream(tmp.outputStream().buffered()).use { out ->
            out.writeInt(a.width)
            out.writeInt(a.height)
            a.landmarks.forEach { out.writeFloat(it) }
            out.write(a.labels)
            val packed = ByteArray((a.person.size + 7) / 8)
            a.person.forEachIndexed { i, on -> if (on) packed[i / 8] = (packed[i / 8].toInt() or (1 shl (i % 8))).toByte() }
            out.write(packed)
        }
        tmp.renameTo(cache)
    }

    private fun analyseFresh(photo: File): PhotoAnalysis {
        val t0 = android.os.SystemClock.uptimeMillis()
        val bitmap = loadUpright(photo, s.workLongSide)
        val t1 = android.os.SystemClock.uptimeMillis()
        val w = bitmap.width
        val h = bitmap.height
        val poseResult = pose.detect(BitmapImageBuilder(bitmap).build())
        val lm = poseResult.landmarks().firstOrNull() ?: throw IllegalStateException("no person found in ${photo.name}")
        val landmarks = FloatArray(33 * 3)
        lm.forEachIndexed { i, p ->
            landmarks[i * 3] = p.x() * w
            landmarks[i * 3 + 1] = p.y() * h
            landmarks[i * 3 + 2] = p.visibility().orElse(0f)
        }
        val maskImage = poseResult.segmentationMasks().get().first()
        var personMat = Mat(maskImage.height, maskImage.width, CvType.CV_32F).apply {
            val floats = ByteBufferExtractor.extract(maskImage).order(ByteOrder.nativeOrder()).asFloatBuffer()
            put(0, 0, FloatArray(floats.remaining()).also { floats.get(it) })
        }
        if (personMat.cols() != w || personMat.rows() != h) {
            val r = Mat()
            Imgproc.resize(personMat, r, Size(w.toDouble(), h.toDouble()))
            personMat.release()
            personMat = r
        }
        val pf = FloatArray(w * h).also { personMat.get(0, 0, it) }
        personMat.release()
        val t2 = android.os.SystemClock.uptimeMillis()

        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        bitmap.recycle()
        val lab = toLab(rgb)
        val t3 = android.os.SystemClock.uptimeMillis()
        val labels = segmenter.labels(rgb)
        val t4 = android.os.SystemClock.uptimeMillis()
        android.util.Log.i("GarmentCutout", "analyse ${photo.name}: decode ${t1 - t0} ms, pose ${t2 - t1} ms, lab ${t3 - t2} ms, clothes model ${t4 - t3} ms")
        return PhotoAnalysis(rgb, landmarks, BooleanArray(w * h) { pf[it] > s.personThreshold }, lab, labels)
    }

    /** Finds the garment pieces in [garment], using the base outfit learned from the body figure. */
    fun pieces(garment: PhotoAnalysis, body: PhotoAnalysis, base: BaseOutfit): Pair<List<PieceResult>, List<DroppedPiece>> {
        val (raw, notes) = extractPieces(garment.lab, garment.labels, base, pieceSettings)
        val n = garment.width * garment.height
        val results = raw.mapNotNull { piece ->
            val clean = cleanup(piece.keep, garment.width, garment.height)
            val kept = clean.count { it }
            if (kept < pieceSettings.minPieceFrac * n) return@mapNotNull null
            val warnings = mutableListOf<String>()
            val (m, info) = try {
                alignToBody(garment.landmarks, body.landmarks, maxOf(body.width, body.height), s, warnings, alignPointsFor(piece.type))
            } catch (e: IllegalStateException) {
                warnings += "${e.message}; placed without alignment"
                dev.wardrobe.app.domain.cutout.Similarity(1.0, 0.0, 0.0, 0.0) to (0 to 0.0)
            }
            val (rgbaInBody, maskInBody) = warpIntoBody(garment.rgb, clean, garment.width, garment.height, m.asRowMajor(), body.width, body.height)
            PieceResult(piece.type, clean, piece.region, piece.baseFrac, maskInBody, rgbaInBody, info.first, info.second, kept.toFloat() / n, warnings)
        }
        return results to notes
    }

    /** Opening (remove specks), closing (fill pinholes), then drop blobs much smaller than the largest. */
    private fun cleanup(mask: BooleanArray, w: Int, h: Int): BooleanArray {
        val m = Mat(h, w, CvType.CV_8U).apply { put(0, 0, ByteArray(w * h) { if (mask[it]) 1 else 0 }) }
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(9.0, 9.0)))
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(m, labels, stats, centroids, 8)
        val out = if (count <= 1) {
            ByteArray(w * h).also { m.get(0, 0, it) }.let { b -> BooleanArray(w * h) { b[it].toInt() != 0 } }
        } else {
            val areas = IntArray(count) { stats.get(it, Imgproc.CC_STAT_AREA)[0].toInt() }
            val largest = (1 until count).maxOf { areas[it] }
            val keep = BooleanArray(count) { it > 0 && areas[it] >= s.minComponentFrac * largest }
            val lab = IntArray(w * h).also { labels.get(0, 0, it) }
            BooleanArray(w * h) { keep[lab[it]] }
        }
        listOf(m, labels, stats, centroids).forEach { it.release() }
        return out
    }

    /** Moves the piece into the body figure's frame: RGBA with a feathered edge, plus a hard mask. */
    private fun warpIntoBody(
        rgb: Mat, mask: BooleanArray, w: Int, h: Int, affine: DoubleArray, bw: Int, bh: Int,
    ): Pair<Bitmap, BooleanArray> {
        val size = Size(bw.toDouble(), bh.toDouble())
        val mat = Mat(2, 3, CvType.CV_64F).apply { put(0, 0, *affine) }
        val alpha = Mat(h, w, CvType.CV_32F).apply { put(0, 0, FloatArray(w * h) { if (mask[it]) 1f else 0f }) }
        Imgproc.GaussianBlur(alpha, alpha, Size(0.0, 0.0), s.featherPx)
        val alphaW = Mat()
        val rgbW = Mat()
        Imgproc.warpAffine(alpha, alphaW, mat, size, Imgproc.INTER_LINEAR)
        Imgproc.warpAffine(rgb, rgbW, mat, size, Imgproc.INTER_LINEAR)
        val a = FloatArray(bw * bh).also { alphaW.get(0, 0, it) }
        val px = ByteArray(bw * bh * 3).also { rgbW.get(0, 0, it) }
        val colors = IntArray(bw * bh) { i ->
            val al = (a[i] * 255f).toInt().coerceIn(0, 255)
            (al shl 24) or ((px[i * 3].toInt() and 0xFF) shl 16) or ((px[i * 3 + 1].toInt() and 0xFF) shl 8) or (px[i * 3 + 2].toInt() and 0xFF)
        }
        listOf(mat, alpha, alphaW, rgbW).forEach { it.release() }
        return Bitmap.createBitmap(colors, bw, bh, Bitmap.Config.ARGB_8888) to BooleanArray(bw * bh) { a[it] >= 0.5f }
    }

    override fun close() {
        pose.close()
        segmenter.close()
    }

    companion object {
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

        /** CIELAB (L 0..100) after a 5x5 blur, like prototype pipeline.to_lab. */
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
    }
}
