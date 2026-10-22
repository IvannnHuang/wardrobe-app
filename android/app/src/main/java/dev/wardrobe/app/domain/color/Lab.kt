package dev.wardrobe.app.domain.color

import kotlin.math.sqrt

// Colour helpers on interleaved CIELAB arrays (L 0..100, a/b roughly -128..127): pixel i is at [3i, 3i+2].

fun deltaE(lab: FloatArray, i: Int, c: FloatArray): Float {
    val dl = lab[i * 3] - c[0]
    val da = lab[i * 3 + 1] - c[1]
    val db = lab[i * 3 + 2] - c[2]
    return sqrt(dl * dl + da * da + db * db)
}

fun deltaE(a: FloatArray, b: FloatArray): Float {
    val dl = a[0] - b[0]
    val da = a[1] - b[1]
    val db = a[2] - b[2]
    return sqrt(dl * dl + da * da + db * db)
}

/** Per-channel median of the masked pixels (sampled to at most [maxSamples]). Null if the mask is empty. */
fun medianLab(lab: FloatArray, mask: BooleanArray, maxSamples: Int = 50_000): FloatArray? {
    val count = mask.count { it }
    if (count == 0) return null
    val step = maxOf(1, count / maxSamples)
    val n = (count + step - 1) / step
    val ch = Array(3) { FloatArray(n) }
    var seen = 0
    var k = 0
    for (i in mask.indices) {
        if (!mask[i]) continue
        if (seen % step == 0 && k < n) {
            for (c in 0..2) ch[c][k] = lab[i * 3 + c]
            k++
        }
        seen++
    }
    return FloatArray(3) { c -> ch[c].also { it.sort(0, k) }[k / 2] }
}

/** Up to [maxSamples] masked pixels as an interleaved LAB array, evenly spread over the mask. */
fun sampleLab(lab: FloatArray, mask: BooleanArray, maxSamples: Int): FloatArray {
    val count = mask.count { it }
    val step = maxOf(1, count / maxSamples)
    val out = ArrayList<Float>(minOf(count, maxSamples) * 3)
    var seen = 0
    for (i in mask.indices) {
        if (!mask[i]) continue
        if (seen % step == 0 && out.size < maxSamples * 3) {
            out.add(lab[i * 3]); out.add(lab[i * 3 + 1]); out.add(lab[i * 3 + 2])
        }
        seen++
    }
    return out.toFloatArray()
}

data class Cluster(val center: FloatArray, val share: Float)

/**
 * k-means on interleaved LAB points (k-means++ start, fixed seed so results are repeatable).
 * Returns clusters sorted by share, largest first.
 */
fun kMeans(points: FloatArray, k: Int, iterations: Int = 30, seed: Long = 0): List<Cluster> {
    val n = points.size / 3
    if (n == 0) return emptyList()
    val kk = minOf(k, n)
    val rnd = java.util.Random(seed)
    val centers = Array(kk) { FloatArray(3) }
    fun p(i: Int) = floatArrayOf(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
    fun d2(i: Int, c: FloatArray): Float {
        val a = points[i * 3] - c[0]
        val b = points[i * 3 + 1] - c[1]
        val d = points[i * 3 + 2] - c[2]
        return a * a + b * b + d * d
    }
    centers[0] = p(rnd.nextInt(n))
    val dist = FloatArray(n) { d2(it, centers[0]) }
    for (c in 1 until kk) {
        val total = dist.sum().toDouble()
        var r = rnd.nextDouble() * total
        var pick = n - 1
        for (i in 0 until n) {
            r -= dist[i]
            if (r <= 0) { pick = i; break }
        }
        centers[c] = p(pick)
        for (i in 0 until n) dist[i] = minOf(dist[i], d2(i, centers[c]))
    }
    val assign = IntArray(n)
    repeat(iterations) {
        var changed = false
        for (i in 0 until n) {
            var best = 0
            var bestD = Float.MAX_VALUE
            for (c in 0 until kk) {
                val d = d2(i, centers[c])
                if (d < bestD) { bestD = d; best = c }
            }
            if (assign[i] != best) { assign[i] = best; changed = true }
        }
        val sums = Array(kk) { DoubleArray(3) }
        val counts = IntArray(kk)
        for (i in 0 until n) {
            val c = assign[i]
            counts[c]++
            for (j in 0..2) sums[c][j] += points[i * 3 + j].toDouble()
        }
        for (c in 0 until kk) if (counts[c] > 0) for (j in 0..2) centers[c][j] = (sums[c][j] / counts[c]).toFloat()
        if (!changed && it > 0) return@repeat
    }
    val counts = IntArray(kk)
    for (i in 0 until n) counts[assign[i]]++
    return (0 until kk).filter { counts[it] > 0 }
        .map { Cluster(centers[it], counts[it].toFloat() / n) }
        .sortedByDescending { it.share }
}
