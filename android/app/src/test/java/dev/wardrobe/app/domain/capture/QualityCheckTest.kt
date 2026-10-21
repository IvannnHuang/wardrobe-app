package dev.wardrobe.app.domain.capture

import dev.wardrobe.app.domain.model.Landmark
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityCheckTest {
    /** Vertical stripes 4 px wide: lots of hard edges. */
    private fun stripes(w: Int, h: Int) = IntArray(w * h) { i -> if ((i % w) / 4 % 2 == 0) 40 else 200 }

    /** Simple box blur, as a stand-in for a blurry photo. */
    private fun blur(src: IntArray, w: Int, h: Int, r: Int): IntArray = IntArray(w * h) { i ->
        val x = i % w
        val y = i / w
        var s = 0
        var n = 0
        for (dy in -r..r) for (dx in -r..r) {
            val xx = (x + dx).coerceIn(0, w - 1)
            val yy = (y + dy).coerceIn(0, h - 1)
            s += src[yy * w + xx]; n++
        }
        s / n
    }

    @Test
    fun sharpBeatsBlurredAndFlatIsZero() {
        val w = 64
        val h = 64
        val sharp = laplacianVariance(stripes(w, h), w, h)
        val soft = laplacianVariance(blur(stripes(w, h), w, h, 3), w, h)
        assertTrue(sharp > soft * 10)
        assertEquals(0.0, laplacianVariance(IntArray(w * h) { 128 }, w, h), 1e-9)
        assertTrue(judgeSharpness(sharp).sharp)
        assertFalse(judgeSharpness(0.0).sharp)
    }

    @Test
    fun personBoxIsPaddedAndClamped() {
        val lm = listOf(
            Landmark(0.40f, 0.10f, 0.9f), Landmark(0.60f, 0.10f, 0.9f),
            Landmark(0.40f, 0.98f, 0.9f), Landmark(0.60f, 0.98f, 0.9f),
            Landmark(0.01f, 0.50f, 0.1f), // hidden: ignored
        )
        assertArrayEquals(floatArrayOf(0.35f, 0.05f, 0.65f, 1f), personBox(lm), 1e-5f)
        assertArrayEquals(floatArrayOf(0f, 0f, 1f, 1f), personBox(emptyList()), 0f)
    }
}
