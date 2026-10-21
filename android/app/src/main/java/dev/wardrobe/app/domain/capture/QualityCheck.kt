package dev.wardrobe.app.domain.capture

import dev.wardrobe.app.domain.model.Landmark

/**
 * Sharpness = variance of the Laplacian over a grayscale image (0..255 values, row-major).
 * Blur removes edges, so the variance drops. Values depend on resolution and scene; calibrate
 * [BLUR_THRESHOLD] from real shots (the score is shown on the last-shot card).
 */
fun laplacianVariance(gray: IntArray, width: Int, height: Int): Double {
    if (width < 3 || height < 3) return 0.0
    var sum = 0.0
    var sumSq = 0.0
    var n = 0
    for (y in 1 until height - 1) {
        val row = y * width
        for (x in 1 until width - 1) {
            val i = row + x
            val lap = (gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width] - 4 * gray[i]).toDouble()
            sum += lap
            sumSq += lap * lap
            n++
        }
    }
    val mean = sum / n
    return sumSq / n - mean * mean
}

/** Shots of a person this soft are obviously blurred (measured on a ~500 px wide downscale). */
const val BLUR_THRESHOLD = 25.0

/**
 * Box around the visible body points, padded, in normalised coordinates (left, top, right, bottom),
 * so sharpness is measured on me and not on a plain wall. Whole image if nobody was seen.
 */
fun personBox(landmarks: List<Landmark>, pad: Float = 0.05f): FloatArray {
    val seen = landmarks.filter { it.visibility >= 0.5f }
    if (seen.size < 4) return floatArrayOf(0f, 0f, 1f, 1f)
    return floatArrayOf(
        (seen.minOf { it.x } - pad).coerceIn(0f, 1f),
        (seen.minOf { it.y } - pad).coerceIn(0f, 1f),
        (seen.maxOf { it.x } + pad).coerceIn(0f, 1f),
        (seen.maxOf { it.y } + pad).coerceIn(0f, 1f),
    )
}

data class QualityResult(val sharpness: Double, val sharp: Boolean)

fun judgeSharpness(sharpness: Double) = QualityResult(sharpness, sharpness >= BLUR_THRESHOLD)
