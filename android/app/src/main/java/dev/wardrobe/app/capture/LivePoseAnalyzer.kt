package dev.wardrobe.app.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import android.util.Size
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import dev.wardrobe.app.domain.model.Landmark
import dev.wardrobe.app.domain.model.PoseFrame
import dev.wardrobe.app.pipeline.pose.PoseDetector
import java.io.Closeable

/**
 * Runs the bundled pose model on low-resolution preview frames, one at a time (frames arriving while
 * the model is busy are dropped). Results are upright, matching what the preview shows.
 */
class LivePoseAnalyzer(
    context: Context,
    private val onPose: (PoseFrame) -> Unit,
) : ImageAnalysis.Analyzer, Closeable {

    private val landmarker = PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(PoseDetector.MODEL_ASSET).build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .build(),
    )

    @Volatile private var closed = false

    override fun analyze(image: ImageProxy) {
        image.use {
            if (closed) return
            val upright = it.toBitmap().rotate(it.imageInfo.rotationDegrees)
            val start = SystemClock.uptimeMillis()
            val result = landmarker.detectForVideo(BitmapImageBuilder(upright).build(), start)
            val landmarks = result.landmarks().firstOrNull().orEmpty().map { p ->
                Landmark(p.x(), p.y(), p.visibility().orElse(0f))
            }
            onPose(PoseFrame(landmarks, upright.width, upright.height, SystemClock.uptimeMillis() - start))
        }
    }

    override fun close() {
        closed = true
        landmarker.close()
    }

    private fun Bitmap.rotate(degrees: Int): Bitmap =
        if (degrees == 0) this
        else Bitmap.createBitmap(this, 0, 0, width, height, Matrix().apply { postRotate(degrees.toFloat()) }, true)

    companion object {
        /** Small 4:3 frames (same framing as the photos), RGBA so they convert straight to a Bitmap. */
        fun buildImageAnalysis(): ImageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                    )
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
    }
}
