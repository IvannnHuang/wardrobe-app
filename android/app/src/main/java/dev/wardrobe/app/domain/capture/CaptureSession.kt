package dev.wardrobe.app.domain.capture

import dev.wardrobe.app.domain.model.Landmark
import dev.wardrobe.app.domain.model.View

enum class CaptureMode { BODY_FIGURE, GARMENT }

/** One photo of the current subject: where it's saved and the pose seen when it was taken. */
data class Shot(val path: String, val landmarks: List<Landmark>)

/**
 * Progress of the current capture: the body figure, or garment N of a session.
 * Pure state; the ViewModel saves files and database rows as it changes.
 */
data class CaptureSession(
    val mode: CaptureMode,
    /** Folder key: the session id for garments, the figure key for the body figure. */
    val key: String,
    /** 1-based garment number within the session (always 1 for the body figure). */
    val garmentNumber: Int = 1,
    val shots: Map<View, Shot> = emptyMap(),
    /** Angle to capture next; null when all 4 are done. */
    val wanted: View? = View.FRONT,
    /** Garments completed in this session so far (including the current one once complete). */
    val garmentsDone: Int = 0,
) {
    val complete get() = wanted == null && shots.size == View.entries.size
    val started get() = shots.isNotEmpty()

    /** Records a photo, then moves on to the first angle still missing (front → right → back → left). */
    fun withShot(view: View, shot: Shot): CaptureSession {
        val newShots = shots + (view to shot)
        // Counts once, when the 4th distinct angle arrives; retakes don't count again.
        val justCompleted = shots.size < View.entries.size && newShots.size == View.entries.size
        return copy(
            shots = newShots,
            wanted = View.entries.firstOrNull { it !in newShots },
            garmentsDone = garmentsDone + if (justCompleted) 1 else 0,
        )
    }

    /** Pick which angle to (re)take next. Retaking replaces that photo. */
    fun choose(view: View) = copy(wanted = view)

    /** Start the next garment of the same session. */
    fun nextGarment() = copy(garmentNumber = garmentNumber + 1, shots = emptyMap(), wanted = View.FRONT)
}

/** Landmarks per view as compact JSON: {"front":[[x,y,visibility],...],...}. Normalised 0..1 coordinates. */
fun landmarksJson(shots: Map<View, Shot>): String =
    View.entries.filter { it in shots }.joinToString(",", "{", "}") { view ->
        shots.getValue(view).landmarks.joinToString(",", "\"${view.key}\":[", "]") {
            "[%.4f,%.4f,%.3f]".format(java.util.Locale.ROOT, it.x, it.y, it.visibility)
        }
    }

/** Raw photo paths per view as JSON: {"front":"captures/...","right":...}. */
fun pathsJson(shots: Map<View, Shot>): String =
    View.entries.filter { it in shots }.joinToString(",", "{", "}") { "\"${it.key}\":\"${shots.getValue(it).path}\"" }

/** Parses a height typed as "175", "175.5" or "175 cm". Returns null if it's not a plausible adult height. */
fun parseHeightCm(text: String): Float? =
    text.trim().removeSuffix("cm").trim().replace(',', '.').toFloatOrNull()?.takeIf { it in 100f..230f }
