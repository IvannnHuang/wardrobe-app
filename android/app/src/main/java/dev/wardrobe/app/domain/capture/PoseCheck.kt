package dev.wardrobe.app.domain.capture

import dev.wardrobe.app.domain.model.Landmark
import dev.wardrobe.app.domain.model.PoseFrame
import dev.wardrobe.app.domain.model.PoseIds
import dev.wardrobe.app.domain.model.View
import kotlin.math.abs

/**
 * Where I should stand, in normalised image coordinates (0..1). Fixed defaults for now; calibrated
 * from the body figure capture later.
 */
data class CaptureTarget(
    val feetX: Float = 0.5f,
    val feetY: Float = 0.92f,
    val feetToleranceX: Float = 0.08f,
    val feetToleranceY: Float = 0.06f,
    val edgeMargin: Float = 0.02f,
)

/** Something that stops the shot, with the hint shown/spoken to me. */
sealed class PoseIssue(val message: String) {
    data object Nobody : PoseIssue("Step into view")
    data object HeadCutOff : PoseIssue("Step back: head is cut off")
    data object FeetCutOff : PoseIssue("Step back: feet are cut off")
    data object SideCutOff : PoseIssue("Move to the centre: body is cut off at the side")
    data object MoveToYourLeft : PoseIssue("Move to your left onto the mark")
    data object MoveToYourRight : PoseIssue("Move to your right onto the mark")
    data object ShuffleForward : PoseIssue("Shuffle forward onto the mark")
    data object ShuffleBackward : PoseIssue("Shuffle backward onto the mark")
    data object MoveCloser : PoseIssue("Move closer to the tablet")
    data object MoveAway : PoseIssue("Move back, away from the tablet")
    data class WrongDirection(val wanted: View, val seen: View?) :
        PoseIssue("Turn to show your ${wanted.spoken()}")
    data object Moving : PoseIssue("Hold still")
}

fun View.spoken(): String = when (this) {
    View.FRONT -> "front"
    View.RIGHT -> "right side"
    View.BACK -> "back"
    View.LEFT -> "left side"
}

data class PoseCheckResult(val issues: List<PoseIssue>, val seenView: View?, val shoulderRatio: Float) {
    val ready get() = issues.isEmpty()
}

private const val MIN_VISIBILITY = 0.5f

// Shoulder width / torso height. Measured on Tab S7+: front ~0.91, sides ~0.22.
private const val FRONT_MIN_RATIO = 0.55f
private const val SIDE_MAX_RATIO = 0.40f

/**
 * Body parts that must be visible for a full-body shot, as left/right pairs. Only one of each pair
 * is required: in a side view the far shoulder, hip and leg are hidden behind the body.
 */
private val REQUIRED_PAIRS = with(PoseIds) {
    listOf(LEFT_SHOULDER to RIGHT_SHOULDER, LEFT_HIP to RIGHT_HIP, LEFT_KNEE to RIGHT_KNEE, LEFT_ANKLE to RIGHT_ANKLE)
}
private val FEET = with(PoseIds) { listOf(LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT, RIGHT_FOOT) }
/** Nose, eyes, ears. Their positions are estimated even from behind, when visibility is low. */
private val HEAD = 0..10

/**
 * Which way I'm facing. The rear camera image is not mirrored, so facing the camera puts my left
 * shoulder on the image's right. Shoulder width relative to torso height separates front/back
 * (wide) from the sides (narrow); face visibility separates front from back; for the sides the nose
 * points the way I'm facing: showing my right side means facing the image's right.
 */
fun detectView(lm: List<Landmark>): Pair<View?, Float> {
    val ls = lm[PoseIds.LEFT_SHOULDER]
    val rs = lm[PoseIds.RIGHT_SHOULDER]
    val torso = abs((lm[PoseIds.LEFT_HIP].y + lm[PoseIds.RIGHT_HIP].y) / 2 - (ls.y + rs.y) / 2)
    if (torso < 1e-3f) return null to 0f
    val ratio = abs(ls.x - rs.x) / torso
    val faceVisible = listOf(0, 2, 5).map { lm[it].visibility }.average() > 0.6
    val view = when {
        ratio >= FRONT_MIN_RATIO -> if (faceVisible && ls.x > rs.x) View.FRONT else if (!faceVisible) View.BACK else null
        ratio <= SIDE_MAX_RATIO -> {
            // In profile the nose sticks out past the ears in the direction I'm facing.
            val facing = lm[PoseIds.NOSE].x - (lm[7].x + lm[8].x) / 2
            when {
                facing > 0.01f -> View.RIGHT
                facing < -0.01f -> View.LEFT
                else -> null
            }
        }
        else -> null // turning
    }
    return view to ratio
}

/** "Still" = key joints moved less than [maxMove] (normalised) over the last [windowMs]. */
class StillnessTracker(private val windowMs: Long = 600, private val maxMove: Float = 0.012f) {
    private val tracked = with(PoseIds) { listOf(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE) }
    private val history = ArrayDeque<Pair<Long, List<Landmark>>>()

    fun reset() = history.clear()

    fun update(lm: List<Landmark>, nowMs: Long): Boolean {
        history.addLast(nowMs to tracked.map { lm[it] })
        while (history.size > 1 && nowMs - history.first().first > windowMs) history.removeFirst()
        if (nowMs - history.first().first < windowMs * 0.8) return false
        return tracked.indices.all { i ->
            val xs = history.map { it.second[i].x }
            val ys = history.map { it.second[i].y }
            (xs.max() - xs.min()) < maxMove && (ys.max() - ys.min()) < maxMove
        }
    }
}

/** Runs all checks for one frame. Keep one instance per camera page (it remembers recent frames). */
class PoseChecker(var target: CaptureTarget = CaptureTarget()) {
    private val stillness = StillnessTracker()

    fun check(frame: PoseFrame, wanted: View, nowMs: Long): PoseCheckResult {
        val lm = frame.landmarks
        if (lm.isEmpty()) {
            stillness.reset()
            return PoseCheckResult(listOf(PoseIssue.Nobody), null, 0f)
        }
        val issues = mutableListOf<PoseIssue>()
        val m = target.edgeMargin

        // 1. Fully in frame (only visible points count; hidden ones are guesses)
        fun seen(i: Int) = lm[i].visibility >= MIN_VISIBILITY
        val upperOk = REQUIRED_PAIRS.take(2).all { (a, b) -> seen(a) || seen(b) }
        val legsOk = REQUIRED_PAIRS.drop(2).all { (a, b) -> seen(a) || seen(b) }
        val headTop = HEAD.minOf { lm[it].y }
        val visibleBody = (11 until lm.size).filter(::seen)
        when {
            !upperOk || headTop < m + 0.05f -> issues += PoseIssue.HeadCutOff
            !legsOk || FEET.any { seen(it) && lm[it].y > 1 - m } -> issues += PoseIssue.FeetCutOff
            visibleBody.any { lm[it].x < m || lm[it].x > 1 - m } -> issues += PoseIssue.SideCutOff
        }

        // 2. On the mark: feet left/right in the image, and feet height (lower in the image = closer)
        val ankles = listOf(PoseIds.LEFT_ANKLE, PoseIds.RIGHT_ANKLE).filter(::seen)
        if (issues.isEmpty() && ankles.isNotEmpty()) {
            val feetX = ankles.map { lm[it].x }.average().toFloat()
            val feetY = ankles.maxOf { lm[it].y }
            when {
                feetX < target.feetX - target.feetToleranceX -> issues += sideways(towardImageRight = true, wanted)
                feetX > target.feetX + target.feetToleranceX -> issues += sideways(towardImageRight = false, wanted)
                feetY < target.feetY - target.feetToleranceY -> issues += PoseIssue.MoveCloser
                feetY > target.feetY + target.feetToleranceY -> issues += PoseIssue.MoveAway
            }
        }

        // 3. Facing direction
        val (seen, ratio) = detectView(lm)
        if (seen != wanted) issues += PoseIssue.WrongDirection(wanted, seen)

        // 4. Still
        if (!stillness.update(lm, nowMs)) issues += PoseIssue.Moving

        return PoseCheckResult(issues, seen, ratio)
    }

    /** Turns "move toward the image's right/left" into words relative to me for the wanted view. */
    private fun sideways(towardImageRight: Boolean, wanted: View): PoseIssue = when (wanted) {
        // Facing the tablet, the image's right is my left.
        View.FRONT -> if (towardImageRight) PoseIssue.MoveToYourLeft else PoseIssue.MoveToYourRight
        View.BACK -> if (towardImageRight) PoseIssue.MoveToYourRight else PoseIssue.MoveToYourLeft
        // Showing my right side I face the image's right, so that way is forward.
        View.RIGHT -> if (towardImageRight) PoseIssue.ShuffleForward else PoseIssue.ShuffleBackward
        View.LEFT -> if (towardImageRight) PoseIssue.ShuffleBackward else PoseIssue.ShuffleForward
    }
}

/** Fires once after the pose has been ready continuously for [holdMs]; then waits [cooldownMs]. */
class AutoShutter(private val holdMs: Long = 500, private val cooldownMs: Long = 2500) {
    private var readySince: Long? = null
    private var lastFired = Long.MIN_VALUE / 2

    fun update(ready: Boolean, nowMs: Long): Boolean {
        if (!ready || nowMs - lastFired < cooldownMs) {
            readySince = null
            return false
        }
        val since = readySince ?: nowMs.also { readySince = it }
        if (nowMs - since >= holdMs) {
            lastFired = nowMs
            readySince = null
            return true
        }
        return false
    }
}
