package dev.wardrobe.app.pipeline.cutout

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * On-device benchmark of the clothes model: speed of each ONNX Runtime provider (and an fp32 model if
 * pushed to /data/local/tmp/clothes_b2_fp32.onnx), and how well its labels agree with the CPU fp16
 * baseline. Results go to logcat (tag ClothesBench). Needs a photo pushed to /data/local/tmp/clothes_bench.jpg
 * (skipped otherwise), so no photo has to live in the project.
 */
@RunWith(AndroidJUnit4::class)
class ClothesModelBenchmarkTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun benchmarkProviders() {
        val photo = File("/data/local/tmp/clothes_bench.jpg")
        assumeTrue("push a photo to ${photo.path}", photo.canRead())
        check(OpenCVLoader.initLocal())
        val bitmap = GarmentCutout.loadUpright(photo, 1600)
        val rgba = Mat().also { Utils.bitmapToMat(bitmap, it) }
        val rgb = Mat().also { Imgproc.cvtColor(rgba, it, Imgproc.COLOR_RGBA2RGB) }
        val w = rgb.cols()
        val h = rgb.rows()

        val variants = mutableListOf<Triple<String, ClothesSegmenter.Provider, File?>>(
            Triple("fp16 CPU", ClothesSegmenter.Provider.CPU, null),
            Triple("fp16 XNNPACK", ClothesSegmenter.Provider.XNNPACK, null),
            Triple("fp16 NNAPI", ClothesSegmenter.Provider.NNAPI, null),
        )
        val fp32 = File("/data/local/tmp/clothes_b2_fp32.onnx")
        if (fp32.canRead()) {
            variants += Triple("fp32 CPU", ClothesSegmenter.Provider.CPU, fp32)
            variants += Triple("fp32 XNNPACK", ClothesSegmenter.Provider.XNNPACK, fp32)
            variants += Triple("fp32 NNAPI", ClothesSegmenter.Provider.NNAPI, fp32)
        }

        var baseline: ByteArray? = null
        for ((name, provider, file) in variants) {
            try {
                val t0 = SystemClock.uptimeMillis()
                ClothesSegmenter(context, provider, file).use { seg ->
                    val load = SystemClock.uptimeMillis() - t0
                    seg.logits(rgb) // warm-up
                    val runs = 3
                    var infer = 0L
                    var up = 0L
                    var labels = ByteArray(0)
                    repeat(runs) {
                        val a = SystemClock.uptimeMillis()
                        val logits = seg.logits(rgb)
                        val b = SystemClock.uptimeMillis()
                        labels = seg.upsample(logits, w, h)
                        infer += b - a
                        up += SystemClock.uptimeMillis() - b
                    }
                    val ref = baseline ?: labels.also { baseline = it }
                    val person = ref.indices.filter { ref[it].toInt() != 0 }
                    val agree = person.count { labels[it] == ref[it] }.toFloat() / maxOf(1, person.size)
                    Log.i(TAG, "%-13s load %5d ms | inference %5d ms | upsample %4d ms | agrees with fp16 CPU %.2f%%"
                        .format(name, load, infer / runs, up / runs, agree * 100))
                }
            } catch (e: Exception) {
                Log.i(TAG, "%-13s FAILED: %s".format(name, e.message?.take(200)))
            }
        }
    }

    private companion object {
        const val TAG = "ClothesBench"
    }
}
