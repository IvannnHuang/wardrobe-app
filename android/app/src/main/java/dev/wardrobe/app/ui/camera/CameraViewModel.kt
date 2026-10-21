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
import dev.wardrobe.app.data.db.BodyFigureDao
import dev.wardrobe.app.data.db.BodyFigureEntity
import dev.wardrobe.app.data.db.CaptureEntryDao
import dev.wardrobe.app.data.db.CaptureEntryEntity
import dev.wardrobe.app.data.files.PhotoStore
import dev.wardrobe.app.domain.capture.CaptureMode
import dev.wardrobe.app.domain.capture.CaptureSession
import dev.wardrobe.app.domain.capture.Shot
import dev.wardrobe.app.domain.capture.landmarksJson
import dev.wardrobe.app.domain.capture.pathsJson
import dev.wardrobe.app.domain.model.Landmark
import dev.wardrobe.app.domain.model.View
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject

/**
 * Camera page state: the body figure setup or garment session in progress, and saving photos.
 * Every photo is written to disk first; a garment goes into the review tray (a CaptureEntry) as soon
 * as all 4 angles exist. Nothing here touches the wardrobe itself.
 */
@HiltViewModel
class CameraViewModel @Inject constructor(
    private val photoStore: PhotoStore,
    private val bodyFigureDao: BodyFigureDao,
    private val captureEntryDao: CaptureEntryDao,
) : ViewModel() {

    data class LastShot(val thumbnail: ImageBitmap, val width: Int, val height: Int, val view: View)

    data class UiState(
        val session: CaptureSession? = null,
        /** Garment N of the session has been put in the review tray. */
        val currentSaved: Boolean = false,
        val capturing: Boolean = false,
        val lastShot: LastShot? = null,
        val hasBodyFigure: Boolean = false,
        val trayCount: Int = 0,
        val message: String? = null,
    ) {
        val askHeight get() = session?.mode == CaptureMode.BODY_FIGURE && session.complete
    }

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            bodyFigureDao.observeCurrent().collect { fig -> _state.update { it.copy(hasBodyFigure = fig != null) } }
        }
        viewModelScope.launch {
            captureEntryDao.observeTrayCount().collect { n -> _state.update { it.copy(trayCount = n) } }
        }
    }

    private fun newKey() = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(Date())

    // ---------------------------------------------------------------- session control

    fun startBodyFigure() = _state.update {
        it.copy(session = CaptureSession(CaptureMode.BODY_FIGURE, newKey()), currentSaved = false, lastShot = null)
    }

    fun startSession() = _state.update {
        it.copy(session = CaptureSession(CaptureMode.GARMENT, newKey()), currentSaved = false, lastShot = null)
    }

    fun choose(view: View) = _state.update { it.copy(session = it.session?.choose(view)) }

    fun nextGarment() = _state.update { s ->
        val session = s.session ?: return@update s
        if (!session.complete) return@update s
        s.copy(session = session.nextGarment(), currentSaved = false, lastShot = null)
    }

    /** Ends the session. A garment with fewer than 4 angles is discarded (its photos are deleted). */
    fun finish() {
        val s = _state.value
        val session = s.session ?: return
        viewModelScope.launch {
            var discarded = false
            if (session.mode == CaptureMode.GARMENT && session.started && !session.complete) {
                withContext(Dispatchers.IO) { photoStore.deleteDir(photoStore.captureDir(session.key, session.garmentNumber)) }
                discarded = true
            }
            if (session.mode == CaptureMode.BODY_FIGURE) {
                withContext(Dispatchers.IO) { photoStore.deleteDir("body/${session.key}") }
            }
            val done = session.garmentsDone
            _state.update {
                it.copy(
                    session = null, currentSaved = false, lastShot = null,
                    message = when (session.mode) {
                        CaptureMode.BODY_FIGURE -> "Body figure setup cancelled"
                        CaptureMode.GARMENT -> "Session finished: $done garment${if (done == 1) "" else "s"} in the review tray" +
                            if (discarded) " (unfinished garment discarded)" else ""
                    },
                )
            }
        }
    }

    fun saveBodyFigure(heightCm: Float) {
        val session = _state.value.session ?: return
        if (session.mode != CaptureMode.BODY_FIGURE || !session.complete) return
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                bodyFigureDao.insert(
                    BodyFigureEntity(
                        frontPath = session.shots.getValue(View.FRONT).path,
                        rightPath = session.shots.getValue(View.RIGHT).path,
                        backPath = session.shots.getValue(View.BACK).path,
                        leftPath = session.shots.getValue(View.LEFT).path,
                        landmarksJson = landmarksJson(session.shots),
                        heightCm = heightCm,
                        createdAt = System.currentTimeMillis(),
                    )
                )
            }
            _state.update { it.copy(session = null, lastShot = null, message = "Body figure saved") }
        }
    }

    // ---------------------------------------------------------------- photos

    fun onCaptureStarted() = _state.update { it.copy(capturing = true) }

    fun onCaptureFailed(message: String) = _state.update { it.copy(capturing = false, message = message) }

    fun messageShown() = _state.update { it.copy(message = null) }

    /** Saves the JPEG for [view] (replacing an earlier take), then records it in the session. */
    fun savePhoto(jpeg: ByteArray, view: View, landmarks: List<Landmark>) {
        val session = _state.value.session ?: return onCaptureFailed("No capture in progress")
        val path = when (session.mode) {
            CaptureMode.BODY_FIGURE -> photoStore.bodyPath(session.key, view)
            CaptureMode.GARMENT -> photoStore.rawCapturePath(session.key, session.garmentNumber, view)
        }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { photoStore.write(path, jpeg) }
                val thumb = withContext(Dispatchers.Default) { decodeThumbnail(jpeg, view) }
                var updated: CaptureSession? = null
                _state.update {
                    updated = it.session?.withShot(view, Shot(path, landmarks))
                    it.copy(session = updated, capturing = false, lastShot = thumb)
                }
                val now = updated ?: return@launch
                if (now.mode == CaptureMode.GARMENT && now.complete && !_state.value.currentSaved) {
                    saveToTray(now)
                }
            } catch (e: Exception) {
                onCaptureFailed("Could not save photo: ${e.message}")
            }
        }
    }

    /** All 4 angles exist: put the garment in the review tray. Later retakes overwrite the same files. */
    private suspend fun saveToTray(session: CaptureSession) {
        withContext(Dispatchers.IO) {
            captureEntryDao.insert(
                CaptureEntryEntity(
                    sessionId = session.key,
                    rawPathsJson = pathsJson(session.shots),
                    createdAt = System.currentTimeMillis(),
                )
            )
        }
        _state.update { it.copy(currentSaved = true) }
    }

    private fun decodeThumbnail(jpeg: ByteArray, view: View): LastShot {
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
        return LastShot(upright.asImageBitmap(), w, h, view)
    }
}
