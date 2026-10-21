package dev.wardrobe.app.domain.cutout

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class CutoutMathTest {

    @Test
    fun similarityRecoversScaleRotationShift() {
        val truth = Similarity(1.1 * cos(0.05), 1.1 * sin(0.05), 12.0, -7.0)
        val src = listOf(doubleArrayOf(100.0, 200.0), doubleArrayOf(300.0, 210.0), doubleArrayOf(120.0, 600.0), doubleArrayOf(280.0, 590.0))
        val dst = src.map { truth.apply(it[0], it[1]) }
        val fit = fitSimilarity(src, dst)
        assertEquals(1.1, fit.scale, 1e-9)
        assertEquals(Math.toDegrees(0.05), fit.rotationDeg, 1e-9)
        assertEquals(12.0, fit.tx, 1e-6)
        assertEquals(-7.0, fit.ty, 1e-6)
    }

    private fun landmarks(shiftX: Float, visible: Float = 0.9f) = FloatArray(33 * 3) { k ->
        val i = k / 3
        when (k % 3) {
            0 -> 400f + (i % 2) * 200f + shiftX
            1 -> 100f + i * 40f
            else -> visible
        }
    }

    @Test
    fun alignToBodyMeasuresShiftAndWarns() {
        val warnings = mutableListOf<String>()
        val (m, info) = alignToBody(landmarks(30f), landmarks(0f), 1600, CutoutSettings(), warnings)
        assertEquals(-30.0, m.tx, 1e-3)
        assertEquals(8, info.first)
        assertEquals(0.0, info.second, 1e-6)
        assertTrue(warnings.isEmpty())
    }

    @Test(expected = IllegalStateException::class)
    fun alignNeedsVisiblePoints() {
        alignToBody(landmarks(0f, visible = 0.1f), landmarks(0f), 1600, CutoutSettings(), mutableListOf())
    }

    @Test
    fun lightingShiftUsesBackgroundMedian() {
        val n = 2000
        val labB = FloatArray(n * 3) { if (it % 3 == 0) 80f else 0f }
        val labG = FloatArray(n * 3) { if (it % 3 == 0) 70f else 0f }
        val warnings = mutableListOf<String>()
        val shift = matchLighting(labG, labB, BooleanArray(n) { true }, warnings)
        assertArrayEquals(floatArrayOf(10f, 0f, 0f), shift, 1e-5f)
        assertEquals(80f, labG[0], 1e-5f)
        assertTrue(warnings.single().startsWith("brightness changed"))
    }

    @Test
    fun changedMaskComparesWithNearestBodyPixel() {
        // pixel 0: same colour as its nearest body pixel; 1: different colour; 2: too far from the body; 3: no body at all
        val labB = floatArrayOf(50f, 0f, 0f, 50f, 0f, 0f, 50f, 0f, 0f, 50f, 0f, 0f)
        val labG = floatArrayOf(52f, 1f, 0f, 20f, 0f, 0f, 50f, 0f, 0f, 50f, 0f, 0f)
        val nearest = intArrayOf(0, 1, 2, -1)
        val dist = floatArrayOf(0f, 0f, 40f, 0f)
        val changed = changedMask(labG, labB, nearest, dist, CutoutSettings())
        assertEquals(listOf(false, true, true, true), changed.toList())
    }
}
