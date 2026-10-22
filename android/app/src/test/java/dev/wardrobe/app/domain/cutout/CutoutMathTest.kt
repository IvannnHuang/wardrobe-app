package dev.wardrobe.app.domain.cutout

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
    fun piecesAlignOnTheirOwnPoints() {
        val warnings = mutableListOf<String>()
        val (m, info) = alignToBody(landmarks(30f), landmarks(0f), 1600, CutoutSettings(), warnings, alignPointsFor(PieceType.SHOES))
        assertEquals(6, info.first)
        assertEquals(-30.0, m.tx, 1e-3)
    }
}
