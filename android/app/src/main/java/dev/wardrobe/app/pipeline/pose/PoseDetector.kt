package dev.wardrobe.app.pipeline.pose

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.io.Closeable

/**
 * Finds the 33 MediaPipe pose landmarks on a still photo, using the model bundled in the APK
 * (assets/models). Runs fully on-device. Not thread-safe: use one instance per worker, and close it.
 */
class PoseDetector(context: Context) : Closeable {

    private val landmarker: PoseLandmarker = PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL_ASSET).build())
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(1) // single user
            .build(),
    )

    fun detect(bitmap: Bitmap): PoseLandmarkerResult =
        landmarker.detect(BitmapImageBuilder(bitmap).build())

    override fun close() = landmarker.close()

    companion object {
        const val MODEL_ASSET = "models/pose_landmarker_full.task"
    }
}
