package dev.wardrobe.app.domain.capture

import dev.wardrobe.app.domain.model.Landmark
import dev.wardrobe.app.domain.model.PoseFrame
import dev.wardrobe.app.domain.model.View
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PoseCheckTest {

    /** A person standing on the default mark, facing the camera (my left shoulder on the image's right). */
    private fun standingFront(): MutableList<Landmark> {
        val p = MutableList(33) { Landmark(0.5f, 0.5f, 0.9f) }
        fun set(i: Int, x: Float, y: Float, v: Float = 0.95f) { p[i] = Landmark(x, y, v) }
        set(0, 0.50f, 0.15f)
        for (i in 1..6) set(i, if (i <= 3) 0.52f else 0.48f, 0.13f)
        set(7, 0.53f, 0.14f); set(8, 0.47f, 0.14f)
        set(11, 0.58f, 0.28f); set(12, 0.42f, 0.28f)
        set(13, 0.60f, 0.42f); set(14, 0.40f, 0.42f)
        set(15, 0.61f, 0.55f); set(16, 0.39f, 0.55f)
        for (i in 17..22) set(i, if (i % 2 == 1) 0.61f else 0.39f, 0.58f)
        set(23, 0.55f, 0.55f); set(24, 0.45f, 0.55f)
        set(25, 0.55f, 0.74f); set(26, 0.45f, 0.74f)
        set(27, 0.55f, 0.90f); set(28, 0.45f, 0.90f)
        set(29, 0.55f, 0.92f); set(30, 0.45f, 0.92f)
        set(31, 0.56f, 0.93f); set(32, 0.44f, 0.93f)
        return p
    }

    private fun frame(lm: List<Landmark>) = PoseFrame(lm, 480, 640, 50)

    /** Feeds the same pose for [ms] milliseconds at 20 fps and returns the last result. */
    private fun hold(checker: PoseChecker, lm: List<Landmark>, wanted: View, ms: Long = 800): PoseCheckResult {
        var result: PoseCheckResult? = null
        for (t in 0..ms step 50) result = checker.check(frame(lm), wanted, t)
        return result!!
    }

    @Test
    fun standingStillOnTheMarkFacingFrontIsReady() {
        val result = hold(PoseChecker(), standingFront(), View.FRONT)
        assertEquals(emptyList<PoseIssue>(), result.issues)
        assertEquals(View.FRONT, result.seenView)
    }

    @Test
    fun movingIsNotReady() {
        val checker = PoseChecker()
        var result: PoseCheckResult? = null
        for (t in 0..800L step 50) {
            val shift = if ((t / 50) % 2 == 0L) 0.03f else 0f
            result = checker.check(frame(standingFront().map { it.copy(x = it.x + shift) }), View.FRONT, t)
        }
        assertTrue(PoseIssue.Moving in result!!.issues)
    }

    @Test
    fun feetCutOffAtTheBottom() {
        val lm = standingFront().apply { for (i in 27..32) this[i] = this[i].copy(y = 0.995f) }
        assertTrue(PoseIssue.FeetCutOff in hold(PoseChecker(), lm, View.FRONT).issues)
    }

    @Test
    fun offTheMarkTellsMeWhichWayToMove() {
        // Standing too far to the image's left while facing the tablet = I need to move to my left.
        val lm = standingFront().map { it.copy(x = it.x - 0.2f) }
        assertTrue(PoseIssue.MoveToYourLeft in hold(PoseChecker(), lm, View.FRONT).issues)
    }

    @Test
    fun backViewIsRecognisedAndRejectedWhenFrontIsWanted() {
        val lm = standingFront().apply {
            for (i in 0..10) this[i] = this[i].copy(visibility = 0.1f)
            val l = this[11]; this[11] = this[12].copy(); this[12] = l
        }
        val result = hold(PoseChecker(), lm, View.FRONT)
        assertEquals(View.BACK, result.seenView)
        assertTrue(result.issues.any { it is PoseIssue.WrongDirection })
    }

    @Test
    fun backViewIsReadyEvenWhenTheModelGuessesAVisibleFace() {
        // From behind: shoulders swapped in the image, but face visibility stays high (seen on the tablet).
        val lm = standingFront().apply { val l = this[11]; this[11] = this[12].copy(); this[12] = l }
        val result = hold(PoseChecker(), lm, View.BACK)
        assertEquals(View.BACK, result.seenView)
        assertEquals(emptyList<PoseIssue>(), result.issues)
    }

    @Test
    fun rightSideIsRecognisedFromNoseDirection() {
        // Showing my right side: shoulders overlap, nose points to the image's right.
        val lm = standingFront().apply {
            this[11] = this[11].copy(x = 0.51f); this[12] = this[12].copy(x = 0.49f)
            this[7] = this[7].copy(x = 0.49f); this[8] = this[8].copy(x = 0.49f)
            this[0] = this[0].copy(x = 0.55f)
        }
        assertEquals(View.RIGHT, hold(PoseChecker(), lm, View.RIGHT).seenView)
        val left = lm.toMutableList().apply { this[0] = this[0].copy(x = 0.43f) }
        assertEquals(View.LEFT, hold(PoseChecker(), left, View.LEFT).seenView)
    }

    @Test
    fun sideViewWithFarLegHiddenIsReady() {
        // Showing my right side: my left shoulder, hip, knee, ankle and foot are behind my body.
        val lm = standingFront().apply {
            this[11] = this[11].copy(x = 0.51f); this[12] = this[12].copy(x = 0.49f)
            this[7] = this[7].copy(x = 0.49f); this[8] = this[8].copy(x = 0.49f)
            this[0] = this[0].copy(x = 0.55f)
            for (i in listOf(11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)) {
                // hidden points: low visibility, and the model's guess may even land off-frame
                this[i] = this[i].copy(y = minOf(this[i].y + 0.1f, 1.02f), visibility = 0.2f)
            }
            for (i in listOf(24, 26, 28, 30, 32)) this[i] = this[i].copy(x = 0.5f)
        }
        val result = hold(PoseChecker(), lm, View.RIGHT)
        assertEquals(emptyList<PoseIssue>(), result.issues)
    }

    @Test
    fun backViewWithFaceHiddenIsReady() {
        val lm = standingFront().apply {
            for (i in 0..10) this[i] = this[i].copy(visibility = 0.05f)
            val l = this[11]; this[11] = this[12].copy(); this[12] = l
        }
        assertEquals(emptyList<PoseIssue>(), hold(PoseChecker(), lm, View.BACK).issues)
    }

    @Test
    fun nobodyInView() {
        assertEquals(listOf(PoseIssue.Nobody), PoseChecker().check(frame(emptyList()), View.FRONT, 0).issues)
    }

    @Test
    fun autoShutterFiresOnceAfterHoldThenCoolsDown() {
        val shutter = AutoShutter(holdMs = 500, cooldownMs = 2500)
        assertFalse(shutter.update(true, 0))
        assertFalse(shutter.update(true, 400))
        assertTrue(shutter.update(true, 500))
        assertFalse(shutter.update(true, 1500))          // cooling down
        assertFalse(shutter.update(false, 3100))         // not ready resets the hold
        assertFalse(shutter.update(true, 3200))
        assertTrue(shutter.update(true, 3700))
    }
}
