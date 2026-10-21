package dev.wardrobe.app.ui.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import dev.wardrobe.app.data.files.PhotoStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
import javax.inject.Inject

/** Holds the Camera page state and saves photos off the main thread. */
@HiltViewModel
class CameraViewModel @Inject constructor(
    private val photoStore: PhotoStore,
) : ViewModel() {

    data class LastShot(val thumbnail: ImageBitmap, val width: Int, val height: Int)

    data class UiState(
        val capturing: Boolean = false,
        val lastShot: LastShot? = null,
        val shotsSaved: Int = 0,
        val error: String? = null,
    )

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    fun onCaptureStarted() = _state.update { it.copy(capturing = true, error = null) }

    fun onCaptureFailed(message: String) = _state.update { it.copy(capturing = false, error = message) }

    fun errorShown() = _state.update { it.copy(error = null) }

    /** Saves the JPEG (with its EXIF rotation) to app-private storage, then shows a thumbnail. */
    fun savePhoto(jpeg: ByteArray) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    photoStore.write(photoStore.manualShotPath(System.currentTimeMillis()), jpeg)
                }
                val shot = withContext(Dispatchers.Default) { decodeThumbnail(jpeg) }
                _state.update { it.copy(capturing = false, lastShot = shot, shotsSaved = it.shotsSaved + 1) }
            } catch (e: Exception) {
                onCaptureFailed("Could not save photo: ${e.message}")
            }
        }
    }

    private fun decodeThumbnail(jpeg: ByteArray): LastShot {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
        val small = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, BitmapFactory.Options().apply { inSampleSize = 8 })
        val degrees = when (ExifInterface(ByteArrayInputStream(jpeg)).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL,
        )) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }
        val upright = if (degrees == 0) small else
            Bitmap.createBitmap(small, 0, 0, small.width, small.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)
        val (w, h) = if (degrees % 180 == 0) bounds.outWidth to bounds.outHeight else bounds.outHeight to bounds.outWidth
        return LastShot(upright.asImageBitmap(), w, h)
    }
}
