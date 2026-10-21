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
import dev.wardrobe.app.domain.capture.judgeSharpness
import dev.wardrobe.app.domain.capture.laplacianVariance
import dev.wardrobe.app.domain.capture.personBox
import dev.wardrobe.app.domain.capture.spoken
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

    /** [view] is null for a close-up. */
    data class LastShot(
        val thumbnail: ImageBitmap,
        val width: Int,
        val height: Int,
        val view: View?,
        val sharpness: Double,
        val sharp: Boolean,
    )

    data class UiState(
        val session: CaptureSession? = null,
        /** Garment N of the session has been put in the review tray (this CaptureEntry). */
        val currentEntryId: Long? = null,
        /** Taking a fabric close-up of the current garment instead of the 4 angles. */
        val closeUpMode: Boolean = false,
        val capturing: Boolean = false,
        val lastShot: LastShot? = null,
        val hasBodyFigure: Boolean = false,
        val trayCount: Int = 0,
        val message: String? = null,
    ) {
        val askHeight get() = session?.mode == CaptureMode.BODY_FIGURE && session.complete
        val currentSaved get() = currentEntryId != null
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
        it.copy(session = CaptureSession(CaptureMode.BODY_FIGURE, newKey()), currentEntryId = null, lastShot = null)
    }

    fun startSession() = _state.update {
        it.copy(session = CaptureSession(CaptureMode.GARMENT, newKey()), currentEntryId = null, lastShot = null)
    }

    fun choose(view: View) = _state.update { it.copy(session = it.session?.choose(view), closeUpMode = false) }

    /** Close-ups are optional and only once all 4 angles of a garment exist. */
    fun startCloseUp() = _state.update {
        if (it.session?.mode == CaptureMode.GARMENT && it.session.complete) it.copy(closeUpMode = true) else it
    }

    fun cancelCloseUp() = _state.update { it.copy(closeUpMode = false) }

    fun nextGarment() = _state.update { s ->
        val session = s.session ?: return@update s
        if (!session.complete) return@update s
        s.copy(session = session.nextGarment(), currentEntryId = null, closeUpMode = false, lastShot = null)
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
                    session = null, currentEntryId = null, closeUpMode = false, lastShot = null,
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
                val thumb = withContext(Dispatchers.Default) { decodeThumbnail(jpeg, view, landmarks) }
                var updated: CaptureSession? = null
                _state.update {
                    // A blurry shot stays on disk, but that angle is queued again for an immediate retake.
                    updated = it.session?.withShot(view, Shot(path, landmarks))
                        ?.let { s -> if (thumb.sharp) s else s.choose(view) }
                    it.copy(
                        session = updated,
                        capturing = false,
                        lastShot = thumb,
                        message = if (thumb.sharp) null else blurryMessage(view, thumb.sharpness),
                    )
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

    private fun blurryMessage(view: View, sharpness: Double) =
        "${view.spoken().replaceFirstChar { it.uppercase() }} was blurry (sharpness ${sharpness.toInt()}), retaking"

    /** All 4 angles exist: put the garment in the review tray. Later retakes overwrite the same files. */
    private suspend fun saveToTray(session: CaptureSession) {
        val id = withContext(Dispatchers.IO) {
            captureEntryDao.insert(
                CaptureEntryEntity(
                    sessionId = session.key,
                    rawPathsJson = pathsJson(session.shots, session.closeUp),
                    createdAt = System.currentTimeMillis(),
                )
            )
        }
        _state.update { it.copy(currentEntryId = id) }
    }

    /** Saves the fabric close-up of the current garment and adds it to its review-tray entry. */
    fun saveCloseUp(jpeg: ByteArray) {
        val s = _state.value
        val session = s.session ?: return onCaptureFailed("No capture in progress")
        val path = photoStore.closeUpPath(session.key, session.garmentNumber)
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { photoStore.write(path, jpeg) }
                val thumb = withContext(Dispatchers.Default) { decodeThumbnail(jpeg, null, emptyList()) }
                val entryId = s.currentEntryId
                if (entryId != null) {
                    withContext(Dispatchers.IO) {
                        captureEntryDao.get(entryId)?.let { entry ->
                            captureEntryDao.update(entry.copy(rawPathsJson = pathsJson(session.shots, path)))
                        }
                    }
                }
                val note = if (thumb.sharp) "Close-up saved"
                else "Close-up saved but looks blurry (sharpness ${thumb.sharpness.toInt()}). Tap Retake close-up."
                _state.update {
                    it.copy(
                        session = it.session?.withCloseUp(path),
                        capturing = false,
                        closeUpMode = false,
                        lastShot = thumb,
                        message = note,
                    )
                }
            } catch (e: Exception) {
                onCaptureFailed("Could not save close-up: ${e.message}")
            }
        }
    }

    /** Small upright copy for the last-shot card, plus the sharpness of the area around me. */
    private fun decodeThumbnail(jpeg: ByteArray, view: View?, landmarks: List<Landmark>): LastShot {
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
        val quality = judgeSharpness(sharpness(upright, personBox(landmarks)))
        return LastShot(upright.asImageBitmap(), w, h, view, quality.sharpness, quality.sharp)
    }

    private fun sharpness(bitmap: Bitmap, box: FloatArray): Double {
        val left = (box[0] * bitmap.width).toInt().coerceIn(0, bitmap.width - 3)
        val top = (box[1] * bitmap.height).toInt().coerceIn(0, bitmap.height - 3)
        val w = ((box[2] - box[0]) * bitmap.width).toInt().coerceIn(3, bitmap.width - left)
        val h = ((box[3] - box[1]) * bitmap.height).toInt().coerceIn(3, bitmap.height - top)
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, left, top, w, h)
        val gray = IntArray(pixels.size) { i ->
            val p = pixels[i]
            (((p shr 16) and 0xFF) * 299 + ((p shr 8) and 0xFF) * 587 + (p and 0xFF) * 114) / 1000
        }
        return laplacianVariance(gray, w, h)
    }
}
