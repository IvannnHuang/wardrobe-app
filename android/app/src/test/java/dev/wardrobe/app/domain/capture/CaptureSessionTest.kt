package dev.wardrobe.app.domain.capture

import dev.wardrobe.app.domain.model.Landmark
import dev.wardrobe.app.domain.model.View
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureSessionTest {
    private fun shot(name: String) = Shot(name, listOf(Landmark(0.5f, 0.25f, 0.9f)))
    private fun session() = CaptureSession(CaptureMode.GARMENT, "s1")

    @Test
    fun anglesGoInOrderAndCompleteAfterFour() {
        var s = session()
        for (view in View.entries) {
            assertEquals(view, s.wanted)
            assertFalse(s.complete)
            s = s.withShot(view, shot(view.key))
        }
        assertTrue(s.complete)
        assertNull(s.wanted)
        assertEquals(1, s.garmentsDone)
    }

    @Test
    fun redoReplacesOnePhotoWithoutCountingTheGarmentTwice() {
        var s = session()
        View.entries.forEach { s = s.withShot(it, shot(it.key)) }

        s = s.choose(View.BACK)
        assertEquals(View.BACK, s.wanted)
        s = s.withShot(View.BACK, shot("back-2"))

        assertTrue(s.complete)
        assertEquals("back-2", s.shots.getValue(View.BACK).path)
        assertEquals(1, s.garmentsDone)
    }

    @Test
    fun skippingAheadComesBackForMissingAngles() {
        var s = session().choose(View.BACK).withShot(View.BACK, shot("b"))
        assertEquals(View.FRONT, s.wanted)
        s = s.withShot(View.FRONT, shot("f"))
        assertEquals(View.RIGHT, s.wanted)
    }

    @Test
    fun nextGarmentStartsFreshAndKeepsCount() {
        var s = session()
        View.entries.forEach { s = s.withShot(it, shot(it.key)) }
        s = s.nextGarment()
        assertEquals(2, s.garmentNumber)
        assertEquals(View.FRONT, s.wanted)
        assertFalse(s.started)
        assertEquals(1, s.garmentsDone)
    }

    @Test
    fun jsonHelpers() {
        val shots = mapOf(View.RIGHT to shot("captures/s1/1/right.jpg"), View.FRONT to shot("captures/s1/1/front.jpg"))
        assertEquals("{\"front\":\"captures/s1/1/front.jpg\",\"right\":\"captures/s1/1/right.jpg\"}", pathsJson(shots))
        assertEquals("{\"front\":[[0.5000,0.2500,0.900]],\"right\":[[0.5000,0.2500,0.900]]}", landmarksJson(shots))
    }

    @Test
    fun heightParsing() {
        assertEquals(175f, parseHeightCm("175"))
        assertEquals(175.5f, parseHeightCm(" 175,5 cm"))
        assertNull(parseHeightCm("5'9"))
        assertNull(parseHeightCm("17"))
    }
}
