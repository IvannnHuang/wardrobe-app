package dev.wardrobe.app.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.wardrobe.app.data.db.BodyFigureDao
import dev.wardrobe.app.data.db.BodyFigureEntity
import dev.wardrobe.app.data.db.CaptureEntryEntity
import dev.wardrobe.app.data.files.PhotoStore
import dev.wardrobe.app.domain.cutout.PieceType
import dev.wardrobe.app.domain.cutout.learnBase
import dev.wardrobe.app.domain.model.View
import dev.wardrobe.app.domain.tags.Measured
import dev.wardrobe.app.pipeline.cutout.GarmentCutout
import dev.wardrobe.app.pipeline.cutout.PhotoAnalysis
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Processes one garment capture: for each angle, find the garment pieces (clothes model, base outfit
 * removed by colour), save each piece as a transparent PNG in the body figure's frame
 * (cutouts/<entryId>/<piece>/<view>.png), and measure tags per piece.
 *
 * cutoutPathsJson: {"pieces": {"top": {"front": {path, alignPoints, alignErrorPx, areaFrac, warnings}, ...}},
 *                   "views": {"front": {"notes": [...], "ms": 1234} | {"error": "..."}}}
 * suggestedTagsJson: {"top": {"colour": {...}, "sleeve": {...}, ...}, "bottom": {...}}
 */
@Singleton
class CaptureProcessor @Inject constructor(
    @ApplicationContext private val context: Context,
    private val photoStore: PhotoStore,
    private val bodyFigureDao: BodyFigureDao,
) {

    data class Output(val cutoutPathsJson: String, val suggestedTagsJson: String)

    suspend fun process(entry: CaptureEntryEntity): Output = withContext(Dispatchers.Default) {
        val body = bodyFigureDao.getCurrent() ?: throw IllegalStateException("no body figure set up")
        val raw = JSONObject(entry.rawPathsJson)
        val pieces = JSONObject()
        val views = JSONObject()
        val tags = mutableMapOf<PieceType, JSONObject>()
        val mainFromFront = mutableSetOf<PieceType>()
        val sideThickness = mutableMapOf<PieceType, MutableList<Measured<Int>>>()

        GarmentCutout(context).use { cutout ->
            // Front first, so its measurements are preferred for the main tags.
            for (view in View.entries) {
                val rawPath = raw.optString(view.key).ifEmpty { null } ?: continue
                val started = System.currentTimeMillis()
                var b: PhotoAnalysis? = null
                var g: PhotoAnalysis? = null
                try {
                    b = cutout.analyse(photoStore.file(body.path(view)))
                    g = cutout.analyse(photoStore.file(rawPath))
                    val base = learnBase(b.lab, b.labels)
                    val (found, notes) = cutout.pieces(g, b, base)
                    for (p in found) {
                        val png = ByteArrayOutputStream().also { p.rgbaInBody.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
                        p.rgbaInBody.recycle()
                        val path = photoStore.write(photoStore.cutoutPath(entry.id, p.type.key, view), png)
                        val perPiece = pieces.optJSONObject(p.type.key) ?: JSONObject().also { pieces.put(p.type.key, it) }
                        perPiece.put(view.key, JSONObject()
                            .put("path", path)
                            .put("alignPoints", p.alignPoints)
                            .put("alignErrorPx", p.alignErrorPx)
                            .put("areaFrac", p.areaFrac.toDouble())
                            .put("baseColourShare", p.baseFrac.toDouble())
                            .put("warnings", JSONArray(p.warnings)))

                        if (p.type !in tags || (view == View.FRONT && p.type !in mainFromFront)) {
                            tags[p.type] = PieceTagger.mainTags(p.type, view, p, g, b)
                            if (view == View.FRONT) mainFromFront += p.type
                        }
                        PieceTagger.thickness(p.type, view, p, b)?.let { sideThickness.getOrPut(p.type) { mutableListOf() } += it }
                    }
                    views.put(view.key, JSONObject().put("notes", JSONArray(notes)).put("ms", System.currentTimeMillis() - started))
                } catch (e: IllegalStateException) {
                    // One bad angle (e.g. nobody detected) shouldn't sink the garment; the review shows the error.
                    Log.w(TAG, "entry ${entry.id} ${view.key}: ${e.message}")
                    views.put(view.key, JSONObject().put("error", e.message))
                } finally {
                    b?.release()
                    g?.release()
                }
            }
        }
        val suggested = JSONObject()
        for ((type, t) in tags) {
            PieceTagger.addThickness(t, sideThickness[type].orEmpty())
            suggested.put(type.key, t)
        }
        Output(JSONObject().put("pieces", pieces).put("views", views).toString(), suggested.toString())
    }

    private fun BodyFigureEntity.path(view: View) = when (view) {
        View.FRONT -> frontPath
        View.RIGHT -> rightPath
        View.BACK -> backPath
        View.LEFT -> leftPath
    }

    private companion object {
        const val TAG = "CaptureProcessor"
    }
}
