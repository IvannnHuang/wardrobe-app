package dev.wardrobe.app.domain.tags

import kotlin.math.hypot

/** Binary image, row-major. Small raster helpers for body regions; pure Kotlin so it's unit-testable. */
class Mask(val width: Int, val height: Int, val data: BooleanArray = BooleanArray(width * height)) {
    init {
        require(data.size == width * height)
    }

    operator fun get(x: Int, y: Int) = data[y * width + x]
    fun count() = data.count { it }
    fun isEmpty() = data.none { it }

    infix fun and(o: Mask) = Mask(width, height, BooleanArray(data.size) { data[it] && o.data[it] })
    infix fun or(o: Mask) = Mask(width, height, BooleanArray(data.size) { data[it] || o.data[it] })
    fun andNot(o: Mask) = Mask(width, height, BooleanArray(data.size) { data[it] && !o.data[it] })

    /** Row indices that contain at least one set pixel. */
    fun rows(): List<Int> = (0 until height).filter { y -> (0 until width).any { this[it, y] } }

    /** All set pixels' y values, sorted (for percentiles). */
    fun ysSorted(): IntArray {
        val out = IntArray(count())
        var k = 0
        for (y in 0 until height) for (x in 0 until width) if (this[x, y]) out[k++] = y
        return out
    }

    companion object {
        fun empty(w: Int, h: Int) = Mask(w, h)

        /** Thick segment with round ends (a "capsule") of radius [r]. */
        fun capsule(w: Int, h: Int, x0: Float, y0: Float, x1: Float, y1: Float, r: Float): Mask {
            val m = Mask(w, h)
            val minX = (minOf(x0, x1) - r).toInt().coerceIn(0, w - 1)
            val maxX = (maxOf(x0, x1) + r).toInt().coerceIn(0, w - 1)
            val minY = (minOf(y0, y1) - r).toInt().coerceIn(0, h - 1)
            val maxY = (maxOf(y0, y1) + r).toInt().coerceIn(0, h - 1)
            val dx = x1 - x0
            val dy = y1 - y0
            val len2 = dx * dx + dy * dy
            for (y in minY..maxY) for (x in minX..maxX) {
                val t = if (len2 == 0f) 0f else (((x - x0) * dx + (y - y0) * dy) / len2).coerceIn(0f, 1f)
                if (hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= r) m.data[y * w + x] = true
            }
            return m
        }

        fun circle(w: Int, h: Int, cx: Float, cy: Float, r: Float) = capsule(w, h, cx, cy, cx, cy, r)

        /** Filled polygon (even-odd rule). */
        fun polygon(w: Int, h: Int, xs: FloatArray, ys: FloatArray): Mask {
            val m = Mask(w, h)
            val minY = ys.min().toInt().coerceIn(0, h - 1)
            val maxY = ys.max().toInt().coerceIn(0, h - 1)
            val minX = xs.min().toInt().coerceIn(0, w - 1)
            val maxX = xs.max().toInt().coerceIn(0, w - 1)
            for (y in minY..maxY) for (x in minX..maxX) {
                var inside = false
                var j = xs.size - 1
                for (i in xs.indices) {
                    if ((ys[i] > y) != (ys[j] > y) && x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]) inside = !inside
                    j = i
                }
                if (inside) m.data[y * w + x] = true
            }
            return m
        }
    }
}

/** Sizes of 8-connected components of [mask]. */
fun componentSizes(mask: Mask): List<Int> {
    val seen = BooleanArray(mask.data.size)
    val sizes = mutableListOf<Int>()
    val stack = IntArray(mask.data.size)
    for (start in mask.data.indices) {
        if (!mask.data[start] || seen[start]) continue
        var top = 0
        stack[top++] = start
        seen[start] = true
        var size = 0
        while (top > 0) {
            val p = stack[--top]
            size++
            val x = p % mask.width
            val y = p / mask.width
            for (dy in -1..1) for (dx in -1..1) {
                val nx = x + dx
                val ny = y + dy
                if (nx < 0 || ny < 0 || nx >= mask.width || ny >= mask.height) continue
                val q = ny * mask.width + nx
                if (mask.data[q] && !seen[q]) { seen[q] = true; stack[top++] = q }
            }
        }
        sizes += size
    }
    return sizes
}
