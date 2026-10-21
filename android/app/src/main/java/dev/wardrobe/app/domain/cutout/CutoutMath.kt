package dev.wardrobe.app.domain.cutout

import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

// Pixel math for the garment cut-out, ported from prototype/pipeline.py. Pure Kotlin so it can be
// unit-tested; image operations (blur, warp, distance transform) live in pipeline/cutout.

/** Shoulders, hips, knees, ankles: the pose landmarks used to align a garment photo to the body figure. */
val ALIGN_POINTS = intArrayOf(11, 12, 23, 24, 25, 26, 27, 28)

/** Selfie multiclass segmenter labels. */
object SegClass {
    const val BACKGROUND = 0
    const val HAIR = 1
    const val BODY_SKIN = 2
    const val FACE_SKIN = 3
    const val CLOTHES = 4
    const val OTHER = 5
}

data class CutoutSettings(
    val minVisibility: Float = 0.6f,
    /** LAB colour difference (Delta E) that counts as "changed" vs the body figure. */
    val changeThreshold: Float = 14f,
    val personThreshold: Float = 0.5f,
    /** Beyond this distance (px) from visible base outfit, person pixels count as garment (hood, wide sleeve). */
    val edgeBandPx: Float = 16f,
    /** Blobs smaller than this fraction of the largest are dropped. */
    val minComponentFrac: Float = 0.02f,
    val featherPx: Double = 1.5,
    /** Long side (px) the photos are processed at. */
    val workLongSide: Int = 1600,
)

/** 2x3 similarity transform [a -b tx; b a ty] (scale + rotation + shift). */
data class Similarity(val a: Double, val b: Double, val tx: Double, val ty: Double) {
    val scale get() = hypot(a, b)
    val rotationDeg get() = Math.toDegrees(atan2(b, a))
    fun apply(x: Double, y: Double) = doubleArrayOf(a * x - b * y + tx, b * x + a * y + ty)
    fun asRowMajor() = doubleArrayOf(a, -b, tx, b, a, ty)
}

/** Least-squares similarity mapping [src] points onto [dst] points (Umeyama, no reflection). */
fun fitSimilarity(src: List<DoubleArray>, dst: List<DoubleArray>): Similarity {
    require(src.size == dst.size && src.size >= 2) { "need at least 2 point pairs" }
    val n = src.size
    val sx = src.sumOf { it[0] } / n
    val sy = src.sumOf { it[1] } / n
    val dx = dst.sumOf { it[0] } / n
    val dy = dst.sumOf { it[1] } / n
    var num1 = 0.0 // sum(xs*xd + ys*yd)
    var num2 = 0.0 // sum(xs*yd - ys*xd)
    var den = 0.0
    for (i in 0 until n) {
        val xs = src[i][0] - sx
        val ys = src[i][1] - sy
        val xd = dst[i][0] - dx
        val yd = dst[i][1] - dy
        num1 += xs * xd + ys * yd
        num2 += xs * yd - ys * xd
        den += xs * xs + ys * ys
    }
    val a = num1 / den
    val b = num2 / den
    return Similarity(a, b, dx - (a * sx - b * sy), dy - (b * sx + a * sy))
}

/** Per-garment-view processing report, stored with the cut-out paths. */
data class CutoutReport(
    val alignPoints: Int,
    val alignScale: Double,
    val alignRotationDeg: Double,
    val alignErrorPx: Double,
    val lightingShiftL: Float,
    val garmentAreaFrac: Float,
    val warnings: List<String>,
)

/**
 * Picks the landmarks visible in both photos and fits the transform from garment photo to body figure.
 * Landmarks are 33x3 (x px, y px, visibility) flattened.
 */
fun alignToBody(
    garment: FloatArray,
    body: FloatArray,
    imageLongSide: Int,
    s: CutoutSettings,
    warnings: MutableList<String>,
): Pair<Similarity, Pair<Int, Double>> {
    val ids = ALIGN_POINTS.filter { garment[it * 3 + 2] >= s.minVisibility && body[it * 3 + 2] >= s.minVisibility }
    if (ids.size < 2) throw IllegalStateException("fewer than 2 body points visible in both photos; cannot align")
    val src = ids.map { doubleArrayOf(garment[it * 3].toDouble(), garment[it * 3 + 1].toDouble()) }
    val dst = ids.map { doubleArrayOf(body[it * 3].toDouble(), body[it * 3 + 1].toDouble()) }
    val m = fitSimilarity(src, dst)
    val error = src.indices.sumOf { i ->
        val p = m.apply(src[i][0], src[i][1])
        hypot(p[0] - dst[i][0], p[1] - dst[i][1])
    } / ids.size
    if (ids.size < 4) warnings += "only ${ids.size} alignment points; alignment may be poor"
    if (error > 0.01 * imageLongSide) warnings += "pose differs from body figure (mean point error %.1f px)".format(error)
    return m to (ids.size to error)
}

/** Median of the values at [mask] for one channel of an interleaved 3-channel array (sampled for speed). */
fun maskedMedian(values: FloatArray, channel: Int, mask: BooleanArray, maxSamples: Int = 50_000): Float {
    val count = mask.count { it }
    if (count == 0) return 0f
    val step = maxOf(1, count / maxSamples)
    val picked = FloatArray((count + step - 1) / step)
    var seen = 0
    var k = 0
    for (i in mask.indices) {
        if (!mask[i]) continue
        if (seen % step == 0 && k < picked.size) picked[k++] = values[i * 3 + channel]
        seen++
    }
    picked.sort(0, k)
    return picked[k / 2]
}

/**
 * Lighting match: shift garment LAB so the shared background (wall) matches the body figure's.
 * Returns the L/a/b shift applied (in place on [labG]).
 */
fun matchLighting(labG: FloatArray, labB: FloatArray, background: BooleanArray, warnings: MutableList<String>): FloatArray {
    if (background.count { it } < 1000) {
        warnings += "too little shared background to match lighting"
        return floatArrayOf(0f, 0f, 0f)
    }
    val shift = FloatArray(3) { c -> maskedMedian(labB, c, background) - maskedMedian(labG, c, background) }
    for (i in background.indices) for (c in 0..2) labG[i * 3 + c] += shift[c]
    if (kotlin.math.abs(shift[0]) > 8) warnings += "brightness changed vs body figure (L shift %.1f)".format(shift[0])
    return shift
}

/**
 * Which pixels differ from the body figure: colour vs the nearest "before" pixel (reliable base outfit),
 * or too far from any such pixel (garment sticking out past the body).
 * [nearest] holds, per pixel, the index of the nearest reliable body-figure pixel (or -1), [dist] its distance.
 */
fun changedMask(
    labG: FloatArray,
    labB: FloatArray,
    nearest: IntArray,
    dist: FloatArray,
    s: CutoutSettings,
): BooleanArray = BooleanArray(nearest.size) { i ->
    val j = nearest[i]
    if (j < 0 || dist[i] > s.edgeBandPx) return@BooleanArray true
    val dl = labG[i * 3] - labB[j * 3]
    val da = labG[i * 3 + 1] - labB[j * 3 + 1]
    val db = labG[i * 3 + 2] - labB[j * 3 + 2]
    sqrt(dl * dl + da * da + db * db) > s.changeThreshold
}
