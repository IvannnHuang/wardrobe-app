package dev.wardrobe.app.pipeline

import dev.wardrobe.app.domain.cutout.ClothesLabel
import dev.wardrobe.app.domain.cutout.PieceType
import dev.wardrobe.app.domain.cutout.maskOf
import dev.wardrobe.app.domain.model.View
import dev.wardrobe.app.domain.tags.Mask
import dev.wardrobe.app.domain.tags.Measured
import dev.wardrobe.app.domain.tags.bodyRegions
import dev.wardrobe.app.domain.tags.bottomsLength
import dev.wardrobe.app.domain.tags.colours
import dev.wardrobe.app.domain.tags.coverage
import dev.wardrobe.app.domain.tags.legShape
import dev.wardrobe.app.domain.tags.legWidths
import dev.wardrobe.app.domain.tags.rise
import dev.wardrobe.app.domain.tags.sheerness
import dev.wardrobe.app.domain.tags.sleeveLength
import dev.wardrobe.app.domain.tags.thickness
import dev.wardrobe.app.domain.tags.topLength
import dev.wardrobe.app.domain.tags.torsoHeight
import dev.wardrobe.app.pipeline.cutout.PhotoAnalysis
import dev.wardrobe.app.pipeline.cutout.PieceResult
import org.json.JSONArray
import org.json.JSONObject

/**
 * Measured tags for one piece. [mainTags] uses one view (front if the piece was found there): colour,
 * coverage, lengths, rise, leg shape, sheerness. [thickness] uses the side views. Called per view so the
 * big per-photo arrays can be released straight away.
 * Output: {"<tag>": {"value": ..., "confidence": 0..1, "measure": number?}}.
 */
object PieceTagger {

    fun mainTags(type: PieceType, view: View, piece: PieceResult, g: PhotoAnalysis, b: PhotoAnalysis): JSONObject {
        val out = JSONObject().put("fromView", view.key)
        val w = g.width
        val h = g.height
        val mask = Mask(w, h, piece.mask)
        val lm = g.landmarks

        val cols = colours(g.lab, piece.mask)
        out.put("colour", JSONObject().put("confidence", 0.85).put("value", JSONArray(cols.map { c ->
            JSONObject().put("name", c.name).put("share", c.share.round(3))
                .put("lab", JSONArray(c.lab.map { it.round(1) })).put("lightness", c.lightness.round(2))
                .put("chroma", c.chroma.round(1)).put("temperature", c.temperature)
        })))

        val frontish = view == View.FRONT || view == View.BACK
        val cov = if (frontish) coverage(mask, bodyRegions(lm, Mask(w, h, g.person))) else emptyMap()
        if (cov.isNotEmpty()) {
            out.put("coverage", JSONObject().put("confidence", 0.7)
                .put("value", JSONObject(cov.mapValues { it.value.round(3) })))
        }
        when (type) {
            PieceType.TOP, PieceType.DRESS -> {
                if (cov.isNotEmpty()) out.putMeasured("sleeve", sleeveLength(cov))
                out.putMeasured("topLength", topLength(mask, lm))
            }
            else -> Unit
        }
        if (type == PieceType.BOTTOM || type == PieceType.DRESS) out.putMeasured("bottomsLength", bottomsLength(mask, lm))
        if (type == PieceType.BOTTOM) {
            val upper = Mask(w, h, intArrayOf(ClothesLabel.UPPER, ClothesLabel.DRESS).maskOf(g.labels))
            out.putMeasured("rise", rise(mask, lm, upper))
            if (frontish) {
                val shape = legShape(legWidths(mask, lm), torsoHeight(lm), legWidths(Mask(b.width, b.height, b.person), b.landmarks), torsoHeight(b.landmarks))
                out.putMeasured("legShape", shape)
            }
        }
        out.putMeasured("sheerness", sheerness(mask, Mask(w, h, piece.region)))
        return out
    }

    /** Side view: how far a top stands off the body. Null for other pieces/views or when not measurable. */
    fun thickness(type: PieceType, view: View, piece: PieceResult, b: PhotoAnalysis): Measured<Int>? {
        if (type != PieceType.TOP && type != PieceType.DRESS) return null
        if (view != View.LEFT && view != View.RIGHT) return null
        return thickness(Mask(b.width, b.height, piece.maskInBody), Mask(b.width, b.height, b.person), b.landmarks)
            .takeIf { it.confidence > 0f }
    }

    /** Averages the side views' thickness into the piece's tags. */
    fun addThickness(tags: JSONObject, sides: List<Measured<Int>>) {
        if (sides.isEmpty()) return
        val level = Math.round(sides.map { it.value }.average()).toInt()
        tags.putMeasured("thickness", Measured(level, sides.first().confidence, sides.mapNotNull { it.measure }.average().toFloat()))
    }

    private fun Float.round(d: Int): Double = Math.round(this * Math.pow(10.0, d.toDouble())) / Math.pow(10.0, d.toDouble())

    private fun JSONObject.putMeasured(key: String, m: Measured<*>) {
        if (m.confidence <= 0f && m.value.toString().startsWith("unknown").not()) return
        put(key, JSONObject().put("value", m.value).put("confidence", m.confidence.round(2)).apply {
            m.measure?.let { put("measure", it.round(3)) }
        })
    }
}
