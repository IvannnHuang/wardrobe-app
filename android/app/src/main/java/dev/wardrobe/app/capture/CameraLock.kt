package dev.wardrobe.app.capture

import android.hardware.camera2.CaptureRequest
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import com.google.common.util.concurrent.ListenableFuture

/**
 * Keeps colours identical across a capture session: exposure (AE) and white balance (AWB) are frozen
 * at their current values until [unlock]. Focus stays automatic so a close-up can refocus.
 */
@OptIn(ExperimentalCamera2Interop::class)
object CameraLock {
    fun lock(camera: Camera): ListenableFuture<Void> =
        Camera2CameraControl.from(camera.cameraControl).setCaptureRequestOptions(
            CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, true)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, true)
                .build()
        )

    fun unlock(camera: Camera): ListenableFuture<Void> =
        Camera2CameraControl.from(camera.cameraControl).clearCaptureRequestOptions()

    /** One-off autofocus on the centre of the frame (for close-ups). */
    fun focusCentre(camera: Camera) =
        camera.cameraControl.startFocusAndMetering(
            FocusMeteringAction.Builder(
                SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.5f),
                FocusMeteringAction.FLAG_AF,
            ).disableAutoCancel().build()
        )
}
