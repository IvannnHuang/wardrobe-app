package dev.wardrobe.app.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.wardrobe.app.data.db.BodyFigureDao
import dev.wardrobe.app.data.db.BodyFigureEntity
import dev.wardrobe.app.data.db.CaptureEntryEntity
import dev.wardrobe.app.data.files.PhotoStore
import dev.wardrobe.app.domain.model.View
import dev.wardrobe.app.pipeline.cutout.Analysis
import dev.wardrobe.app.pipeline.cutout.GarmentCutout
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Runs the processing steps for one garment: for each captured angle, cut the garment out against the
 * body figure's same angle and save it as a transparent PNG (cutouts/<entryId>/<view>.png).
 */
@Singleton
class CaptureProcessor @Inject constructor(
    @ApplicationContext private val context: Context,
    private val photoStore: PhotoStore,
    private val bodyFigureDao: BodyFigureDao,
) {

    data class Output(val cutoutPathsJson: String?)

    suspend fun process(entry: CaptureEntryEntity): Output = withContext(Dispatchers.Default) {
        val body = bodyFigureDao.getCurrent() ?: throw IllegalStateException("no body figure set up")
        val raw = JSONObject(entry.rawPathsJson)
        val result = JSONObject()
        GarmentCutout(context).use { cutout ->
            for (view in View.entries) {
                val rawPath = raw.optString(view.key).ifEmpty { null } ?: continue
                val started = System.currentTimeMillis()
                var bodyAnalysis: Analysis? = null
                var garmentAnalysis: Analysis? = null
                try {
                    bodyAnalysis = cutout.analyse(photoStore.file(body.path(view)))
                    garmentAnalysis = cutout.analyse(photoStore.file(rawPath))
                    val out = cutout.cutOut(garmentAnalysis, bodyAnalysis)
                    val png = ByteArrayOutputStream().also { out.rgba.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
                    out.rgba.recycle()
                    val path = photoStore.write(photoStore.cutoutPath(entry.id, view), png)
                    val r = out.report
                    result.put(view.key, JSONObject()
                        .put("path", path)
                        .put("alignPoints", r.alignPoints)
                        .put("alignScale", r.alignScale)
                        .put("alignRotationDeg", r.alignRotationDeg)
                        .put("alignErrorPx", r.alignErrorPx)
                        .put("lightingShiftL", r.lightingShiftL.toDouble())
                        .put("areaFrac", r.garmentAreaFrac.toDouble())
                        .put("warnings", JSONArray(r.warnings))
                        .put("ms", System.currentTimeMillis() - started))
                } catch (e: IllegalStateException) {
                    // One bad angle (e.g. nobody detected) shouldn't sink the garment; review shows the error.
                    Log.w(TAG, "entry ${entry.id} ${view.key}: ${e.message}")
                    result.put(view.key, JSONObject().put("error", e.message))
                } finally {
                    bodyAnalysis?.release()
                    garmentAnalysis?.release()
                }
            }
        }
        Output(result.toString())
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
