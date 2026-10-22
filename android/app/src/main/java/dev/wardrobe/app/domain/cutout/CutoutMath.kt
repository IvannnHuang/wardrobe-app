package dev.wardrobe.app.domain.cutout

import kotlin.math.atan2
import kotlin.math.hypot

// Aligning a garment piece onto the body figure: a similarity transform (scale + rotation + shift)
// fitted to matching pose landmarks. Pure Kotlin so it can be unit-tested.

/** Shoulders, hips, knees, ankles: default alignment points. */
val ALIGN_POINTS = intArrayOf(11, 12, 23, 24, 25, 26, 27, 28)

/** Landmarks to align each piece with (CLAUDE.md: tops shoulders + hips, bottoms hips to ankles, shoes ankles). */
fun alignPointsFor(piece: PieceType): IntArray = when (piece) {
    PieceType.TOP, PieceType.SCARF, PieceType.BAG -> intArrayOf(11, 12, 23, 24)
    PieceType.DRESS -> intArrayOf(11, 12, 23, 24, 25, 26)
    PieceType.BOTTOM, PieceType.BELT -> intArrayOf(23, 24, 25, 26, 27, 28)
    PieceType.SHOES -> intArrayOf(27, 28, 29, 30, 31, 32)
    PieceType.HAT -> intArrayOf(0, 7, 8, 11, 12)
}

data class CutoutSettings(
    val minVisibility: Float = 0.6f,
    val personThreshold: Float = 0.5f,
    /** Blobs smaller than this fraction of the piece's largest blob are dropped. */
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
    points: IntArray = ALIGN_POINTS,
): Pair<Similarity, Pair<Int, Double>> {
    val ids = points.filter { garment[it * 3 + 2] >= s.minVisibility && body[it * 3 + 2] >= s.minVisibility }
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
