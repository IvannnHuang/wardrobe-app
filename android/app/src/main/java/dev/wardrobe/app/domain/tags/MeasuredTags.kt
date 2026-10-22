package dev.wardrobe.app.domain.tags

import dev.wardrobe.app.domain.color.deltaE
import dev.wardrobe.app.domain.color.kMeans
import dev.wardrobe.app.domain.color.sampleLab
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

// Measured tags (Phase 2.3), ported from prototype/tags.py. Measurements are made in the garment photo's
// own frame with that photo's landmarks (px, 33 x [x, y, visibility]) and person mask, so they don't
// depend on matching the body figure's pose. Confidences marked LOW are heuristics awaiting test photos.

data class Measured<T>(val value: T, val confidence: Float, val measure: Float? = null)

// ---------------------------------------------------------------- colour

data class NamedColour(
    val name: String,
    val share: Float,
    val lab: FloatArray,
    val lightness: Float,
    val chroma: Float,
    val temperature: String,
)

val COLOUR_NAMES: Map<String, FloatArray> = mapOf(
    "black" to floatArrayOf(12f, 0f, 0f), "charcoal" to floatArrayOf(30f, 0f, -2f), "grey" to floatArrayOf(55f, 0f, -1f),
    "light grey" to floatArrayOf(78f, 0f, -1f), "white" to floatArrayOf(96f, 0f, 1f), "cream" to floatArrayOf(92f, 0f, 12f),
    "beige" to floatArrayOf(80f, 4f, 18f), "tan" to floatArrayOf(66f, 10f, 28f), "brown" to floatArrayOf(38f, 16f, 26f),
    "camel" to floatArrayOf(62f, 14f, 38f), "olive" to floatArrayOf(48f, -10f, 36f), "khaki" to floatArrayOf(70f, -2f, 28f),
    "green" to floatArrayOf(50f, -45f, 30f), "forest green" to floatArrayOf(32f, -28f, 14f), "mint" to floatArrayOf(88f, -22f, 8f),
    "teal" to floatArrayOf(48f, -28f, -8f), "navy" to floatArrayOf(20f, 10f, -36f), "blue" to floatArrayOf(45f, 15f, -60f),
    "light blue" to floatArrayOf(78f, -8f, -22f), "denim" to floatArrayOf(45f, -2f, -24f), "purple" to floatArrayOf(38f, 40f, -40f),
    "lavender" to floatArrayOf(78f, 14f, -18f), "burgundy" to floatArrayOf(28f, 38f, 12f), "red" to floatArrayOf(50f, 68f, 48f),
    "pink" to floatArrayOf(72f, 32f, 2f), "orange" to floatArrayOf(65f, 40f, 64f), "mustard" to floatArrayOf(68f, 8f, 62f),
    "yellow" to floatArrayOf(88f, -6f, 80f),
)

fun colourName(lab: FloatArray): String = COLOUR_NAMES.minBy { deltaE(it.value, lab) }.key

/** Main colours of the piece: k-means in LAB, shades merged (< 10 Delta E, then same name), share >= 5%. */
fun colours(lab: FloatArray, mask: BooleanArray, k: Int = 4, minShare: Float = 0.05f): List<NamedColour> {
    if (mask.count { it } < 50) return emptyList()
    val clusters = kMeans(sampleLab(lab, mask, 30_000), k)
    val merged = mutableListOf<Pair<FloatArray, Float>>()
    fun mergeInto(list: MutableList<Pair<FloatArray, Float>>, c: FloatArray, s: Float, same: (FloatArray) -> Boolean) {
        val i = list.indexOfFirst { same(it.first) }
        if (i < 0) list += c.copyOf() to s
        else {
            val (mc, ms) = list[i]
            list[i] = FloatArray(3) { (mc[it] * ms + c[it] * s) / (ms + s) } to (ms + s)
        }
    }
    for ((c, s) in clusters) mergeInto(merged, c, s) { deltaE(it, c) < 10f }
    val named = mutableListOf<Pair<FloatArray, Float>>()
    for ((c, s) in merged) mergeInto(named, c, s) { colourName(it) == colourName(c) }
    return named.sortedByDescending { it.second }.filter { it.second >= minShare }.map { (c, s) ->
        val chroma = hypot(c[1], c[2])
        val hue = ((Math.toDegrees(atan2(c[2], c[1]).toDouble()) + 360) % 360).toFloat()
        val temp = if (chroma < 10f) "neutral" else if (hue < 110f || hue > 330f) "warm" else "cool"
        NamedColour(colourName(c), s, c, c[0] / 100f, chroma, temp)
    }
}

// ---------------------------------------------------------------- body regions + coverage

private fun FloatArray.x(i: Int) = this[i * 3]
private fun FloatArray.y(i: Int) = this[i * 3 + 1]
private fun FloatArray.vis(i: Int) = this[i * 3 + 2]

fun torsoHeight(lm: FloatArray): Float =
    hypot((lm.x(11) + lm.x(12)) / 2 - (lm.x(23) + lm.x(24)) / 2, (lm.y(11) + lm.y(12)) / 2 - (lm.y(23) + lm.y(24)) / 2)

/** Body regions from landmarks, limited to the person mask. Hidden limbs (side views) are skipped. */
fun bodyRegions(lm: FloatArray, person: Mask, minVis: Float = 0.5f): Map<String, Mask> {
    val w = person.width
    val h = person.height
    val t = torsoHeight(lm)
    fun visible(vararg ids: Int) = ids.all { lm.vis(it) >= minVis }
    val regions = mutableMapOf<String, Mask>()
    fun add(name: String, m: Mask) {
        val clipped = m and person
        regions[name] = regions[name]?.or(clipped) ?: clipped
    }
    fun capsule(a: Int, b: Int, r: Float) = Mask.capsule(w, h, lm.x(a), lm.y(a), lm.x(b), lm.y(b), r)

    val torsoIds = intArrayOf(11, 12, 24, 23)
    // The plain shoulder-hip quad: arms are measured outside it, so a sleeveless top's shoulders don't count as sleeve.
    val torsoQuad = Mask.polygon(w, h, FloatArray(4) { lm.x(torsoIds[it]) }, FloatArray(4) { lm.y(torsoIds[it]) })
    if (torsoIds.count { lm.vis(it) >= minVis } >= 3) {
        val cx = torsoIds.map { lm.x(it) }.average().toFloat()
        add("torso", Mask.polygon(w, h, FloatArray(4) { cx + (lm.x(torsoIds[it]) - cx) * 1.15f }, FloatArray(4) { lm.y(torsoIds[it]) }))
    }
    if (visible(0)) add("head", Mask.circle(w, h, lm.x(0), lm.y(0), 0.35f * t))
    var knees = Mask.empty(w, h)
    for (knee in intArrayOf(25, 26)) if (visible(knee)) knees = knees or Mask.circle(w, h, lm.x(knee), lm.y(knee), 0.14f * t)
    if (!knees.isEmpty()) add("knees", knees)
    for (side in listOf(intArrayOf(11, 13, 15, 23, 25, 27, 29, 31), intArrayOf(12, 14, 16, 24, 26, 28, 30, 32))) {
        val (s, e, wr, hip, knee) = side.toList()
        val ankle = side[5]; val heel = side[6]; val toe = side[7]
        if (visible(s, e)) add("upperArms", capsule(s, e, 0.16f * t).andNot(torsoQuad))
        if (visible(e, wr)) add("forearms", capsule(e, wr, 0.13f * t).andNot(torsoQuad))
        if (visible(hip, knee)) add("thighs", capsule(hip, knee, 0.22f * t).andNot(knees))
        if (visible(knee, ankle)) add("lowerLegs", capsule(knee, ankle, 0.15f * t).andNot(knees))
        if (visible(heel, toe)) add("feet", capsule(heel, toe, 0.10f * t))
    }
    val arms = listOfNotNull(regions["upperArms"], regions["forearms"]).fold(Mask.empty(w, h)) { a, b -> a or b }
    regions["torso"]?.let { regions["torso"] = it.andNot(arms) }
    return regions
}

/** Share of each body region the garment covers (regions under 200 px are skipped as unreliable). */
fun coverage(garment: Mask, regions: Map<String, Mask>): Map<String, Float> =
    regions.filterValues { it.count() > 200 }.mapValues { (_, r) -> (garment and r).count().toFloat() / r.count() }

// ---------------------------------------------------------------- lengths, rise, legs

fun sleeveLength(cov: Map<String, Float>): Measured<String> {
    val up = cov["upperArms"] ?: return Measured("unknown", 0f)
    val fo = cov["forearms"] ?: 0f
    return when {
        fo > 0.75f -> Measured("long", 0.9f)
        fo > 0.35f -> Measured("three-quarter", 0.7f)
        up > 0.45f -> Measured("short", 0.8f)
        up > 0.15f -> Measured("cap", 0.6f)
        else -> Measured("sleeveless", 0.8f)
    }
}

/** Vertical position of [y]: 0 at the mid-point of landmarks a, 1 at the mid-point of b. */
fun relativeY(y: Float, lm: FloatArray, a: Pair<Int, Int>, b: Pair<Int, Int>): Float {
    val ya = (lm.y(a.first) + lm.y(a.second)) / 2
    val yb = (lm.y(b.first) + lm.y(b.second)) / 2
    val span = yb - ya
    return if (abs(span) < 1f) 0f else (y - ya) / span
}

private fun IntArray.percentile(p: Float) = this[((size - 1) * p).toInt().coerceIn(0, size - 1)].toFloat()

fun topLength(garment: Mask, lm: FloatArray): Measured<String> {
    val ys = garment.ysSorted()
    if (ys.isEmpty()) return Measured("unknown", 0f)
    val t = relativeY(ys.percentile(0.99f), lm, 23 to 24, 25 to 26) // 0 hip, 1 knee
    val name = when {
        t < -0.25f -> "cropped"
        t < -0.05f -> "waist"
        t < 0.25f -> "hip"
        t < 0.6f -> "long"
        else -> "tunic"
    }
    return Measured(name, 0.75f, t)
}

fun bottomsLength(garment: Mask, lm: FloatArray): Measured<String> {
    val ys = garment.ysSorted()
    if (ys.isEmpty()) return Measured("unknown", 0f)
    val t = relativeY(ys.percentile(0.99f), lm, 25 to 26, 27 to 28) // 0 knee, 1 ankle
    val name = when {
        t < -0.6f -> "short"
        t < -0.15f -> "above-knee"
        t < 0.2f -> "knee"
        t < 0.7f -> "midi"
        t < 0.95f -> "ankle"
        else -> "full"
    }
    return Measured(name, 0.75f, t)
}

/** True if a top (garment or base top) sits directly above the bottom piece's upper edge. */
fun waistbandCovered(bottom: Mask, upper: Mask, band: Int = 12): Boolean {
    var cols = 0
    var hits = 0
    for (x in 0 until bottom.width) {
        val top = (0 until bottom.height).firstOrNull { bottom[x, it] } ?: continue
        cols++
        if ((maxOf(0, top - band) until top).any { upper[x, it] }) hits++
    }
    return cols > 0 && hits.toFloat() / cols > 0.5f
}

/** Waistband height: 0 at the hips, 1 at the shoulders. Needs the waistband visible. LOW confidence. */
fun rise(bottom: Mask, lm: FloatArray, upper: Mask?): Measured<String> {
    val ys = bottom.ysSorted()
    if (ys.isEmpty()) return Measured("unknown", 0f)
    if (upper != null && waistbandCovered(bottom, upper)) return Measured("unknown (waistband covered)", 0f)
    val r = -relativeY(ys.percentile(0.01f), lm, 23 to 24, 11 to 12)
    val name = if (r < 0.08f) "low" else if (r < 0.22f) "mid" else "high"
    return Measured(name, 0.4f, r)
}

/** Width of the run on row [y] containing (or nearest to) column [x], only counting columns in [from, to). */
fun runWidth(mask: Mask, y: Int, x: Float, from: Int = 0, to: Int = mask.width): Int =
    run(mask, y, x, from, to)?.let { it.last - it.first + 1 } ?: 0

/** The run on row [y] containing (or nearest to) column [x], within [from, to). */
fun run(mask: Mask, y: Int, x: Float, from: Int = 0, to: Int = mask.width): IntRange? {
    var best: IntRange? = null
    var bestDist = Float.MAX_VALUE
    var start = -1
    val lo = from.coerceIn(0, mask.width)
    val hi = to.coerceIn(lo, mask.width)
    for (cx in lo..hi) {
        val on = cx < hi && mask[cx, y]
        if (on && start < 0) start = cx
        if (!on && start >= 0) {
            val end = cx - 1
            val d = if (x >= start && x <= end) 0f else minOf(abs(start - x), abs(end - x))
            if (d < bestDist) { bestDist = d; best = start..end }
            start = -1
        }
    }
    return best
}

/**
 * Mean left/right leg width (px) at thigh, knee, calf and ankle levels. Each leg is measured only on its
 * own side of the midline between the legs, so legs standing close together aren't counted as one.
 */
fun legWidths(mask: Mask, lm: FloatArray): Map<String, Float> {
    val levels = mapOf(
        "thigh" to Triple(23 to 25, 24 to 26, 0.3f), "knee" to Triple(25 to 25, 26 to 26, 0f),
        "calf" to Triple(25 to 27, 26 to 28, 0.5f), "ankle" to Triple(27 to 27, 28 to 28, 0f),
    )
    val out = mutableMapOf<String, Float>()
    for ((name, level) in levels) {
        fun point(p: Pair<Int, Int>) = floatArrayOf(
            lm.x(p.first) + (lm.x(p.second) - lm.x(p.first)) * level.third,
            lm.y(p.first) + (lm.y(p.second) - lm.y(p.first)) * level.third,
        )
        val pa = point(level.first)
        val pb = point(level.second)
        val mid = ((pa[0] + pb[0]) / 2).toInt()
        val bothVisible = listOf(level.first, level.second).all { (a, b) -> lm.vis(a) >= 0.5f && lm.vis(b) >= 0.5f }
        val widths = listOf(level.first to pa, level.second to pb).mapNotNull { (ids, p) ->
            if (lm.vis(ids.first) < 0.5f || lm.vis(ids.second) < 0.5f) return@mapNotNull null
            val py = p[1].toInt()
            if (py !in 0 until mask.height) return@mapNotNull null
            val (from, to) = when {
                !bothVisible -> 0 to mask.width
                p[0] < mid -> 0 to mid
                else -> mid to mask.width
            }
            val r = run(mask, py, p[0], from, to) ?: return@mapNotNull null
            // Cut off at the midline (wide legs touching): assume the leg is symmetric around its centre.
            when {
                bothVisible && r.first == from && from > 0 -> 2 * (r.last - p[0]).coerceAtLeast(1f)
                bothVisible && r.last == to - 1 && to < mask.width -> 2 * (p[0] - r.first).coerceAtLeast(1f)
                else -> (r.last - r.first + 1).toFloat()
            }
        }
        if (widths.isNotEmpty()) out[name] = widths.average().toFloat()
    }
    return out
}

/** Garment leg widths vs my legs, each divided by its own photo's torso height. LOW confidence. */
fun legShape(garment: Map<String, Float>, garmentTorso: Float, body: Map<String, Float>, bodyTorso: Float): Measured<String> {
    val ratios = garment.filterKeys { (body[it] ?: 0f) > 0f }.mapValues { (k, v) -> (v / garmentTorso) / (body.getValue(k) / bodyTorso) }
    val knee = ratios["knee"]
    val ankle = ratios["ankle"]
    if (knee == null || ankle == null) return Measured("unknown", 0f)
    val thigh = ratios["thigh"] ?: knee
    val taper = garment.getValue("ankle") / garment.getValue("knee")
    val name = when {
        taper > 1.15f && ankle > 1.3f -> "flare"
        knee < 1.15f && ankle < 1.15f -> "skinny"
        knee > 1.5f && ankle > 1.5f -> "wide"
        taper < 0.8f && thigh > 1.2f -> "tapered"
        else -> "straight"
    }
    return Measured(name, 0.45f, knee)
}

/** Side view, in the body figure's frame: how far the garment stands off my body outline. LOW confidence. */
fun thickness(garmentInBody: Mask, bodyPerson: Mask, bodyLm: FloatArray): Measured<Int> {
    val t = torsoHeight(bodyLm)
    val y0 = (minOf(bodyLm.y(11), bodyLm.y(12)) + 0.2f * t).toInt().coerceAtLeast(0)
    val y1 = (maxOf(bodyLm.y(23), bodyLm.y(24)) - 0.1f * t).toInt().coerceAtMost(bodyPerson.height)
    val gaps = mutableListOf<Float>()
    for (y in y0 until y1) {
        val g = (0 until garmentInBody.width).filter { garmentInBody[it, y] }
        val b = (0 until bodyPerson.width).filter { bodyPerson[it, y] }
        if (g.size > 5 && b.size > 5) gaps += ((g.last() - g.first()) - (b.last() - b.first())) / 2f
    }
    if (gaps.isEmpty()) return Measured(0, 0f)
    val gap = gaps.sorted()[gaps.size / 2] / t
    val level = if (gap < 0.02f) 1 else if (gap < 0.04f) 2 else if (gap < 0.07f) 3 else if (gap < 0.1f) 4 else 5
    return Measured(level, 0.35f, gap)
}

/** Base colour showing through as small holes spread inside the piece (big holes = openings). LOW. */
fun sheerness(keep: Mask, region: Mask): Measured<String> {
    val area = maxOf(1, region.count())
    val small = componentSizes(region.andNot(keep)).filter { it < 0.01f * area }.sum()
    val share = small.toFloat() / area
    val name = if (share < 0.03f) "opaque" else if (share < 0.12f) "semi-sheer" else "sheer"
    return Measured(name, 0.3f, share)
}
