package dev.wardrobe.app.pipeline.cutout

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * Clothes segmentation (SegFormer-B2 trained on ATR, 18 labels) via ONNX Runtime, fully on-device.
 * Same preprocessing as prototype/clothes.py: 512x512, ImageNet mean/std; logits are upsampled to the
 * photo size per label and the best label per pixel is kept. Not thread-safe; close when done.
 */
class ClothesSegmenter(context: Context) : Closeable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // ONNX Runtime needs a file path; copy the model out of the APK once (55 MB) instead of loading
        // it onto the Java heap.
        val file = File(context.noBackupFilesDir, MODEL_ASSET.substringAfterLast('/'))
        val assetSize = context.assets.openFd(MODEL_ASSET).use { it.length }
        if (!file.exists() || file.length() != assetSize) {
            val tmp = File(file.path + ".tmp")
            context.assets.open(MODEL_ASSET).use { input -> tmp.outputStream().use { input.copyTo(it) } }
            tmp.renameTo(file)
        }
        session = env.createSession(
            file.path,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            },
        )
    }

    /** Label per pixel (row-major, values 0..17) for an RGB CV_8UC3 image. */
    fun labels(rgb: Mat): ByteArray {
        val w = rgb.cols()
        val h = rgb.rows()
        val small = Mat()
        Imgproc.resize(rgb, small, Size(SIZE.toDouble(), SIZE.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val px = ByteArray(SIZE * SIZE * 3).also { small.get(0, 0, it) }
        small.release()
        val input = FloatArray(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) for (c in 0..2) {
            input[c * plane + i] = ((px[i * 3 + c].toInt() and 0xFF) / 255f - MEAN[c]) / STD[c]
        }
        val logits = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            session.run(mapOf(INPUT to tensor)).use { result ->
                val out = result[0] as OnnxTensor
                val shape = out.info.shape // 1 x 18 x 128 x 128
                val buf = out.floatBuffer
                Triple(FloatArray(buf.remaining()).also { buf.get(it) }, shape[2].toInt(), shape[3].toInt())
            }
        }
        val (data, lh, lw) = logits
        val classes = data.size / (lh * lw)

        // Upsample one label's logits at a time and keep the running best, so memory stays small.
        val best = Mat(h, w, CvType.CV_32F, Scalar(-1e9))
        val label = Mat(h, w, CvType.CV_8U, Scalar(0.0))
        val low = Mat(lh, lw, CvType.CV_32F)
        val up = Mat()
        val better = Mat()
        for (c in 0 until classes) {
            low.put(0, 0, data.copyOfRange(c * lh * lw, (c + 1) * lh * lw))
            Imgproc.resize(low, up, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Core.compare(up, best, better, Core.CMP_GT)
            up.copyTo(best, better)
            label.setTo(Scalar(c.toDouble()), better)
        }
        val out = ByteArray(w * h).also { label.get(0, 0, it) }
        listOf(best, label, low, up, better).forEach { it.release() }
        return out
    }

    override fun close() = session.close()

    companion object {
        const val MODEL_ASSET = "models/clothes_segformer_b2_fp16.onnx"
        private const val INPUT = "pixel_values"
        private const val SIZE = 512
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
