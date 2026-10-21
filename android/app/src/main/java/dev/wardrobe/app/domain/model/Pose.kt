package dev.wardrobe.app.domain.model

/** One MediaPipe pose landmark. x and y are 0..1 across the upright image; visibility is 0..1. */
data class Landmark(val x: Float, val y: Float, val visibility: Float)

/** Pose found in one camera frame (landmarks empty when nobody is in view). */
data class PoseFrame(
    val landmarks: List<Landmark>,
    val imageWidth: Int,
    val imageHeight: Int,
    val inferenceMs: Long,
)

/** MediaPipe landmark ids used in the app. */
object PoseIds {
    const val NOSE = 0
    const val LEFT_SHOULDER = 11
    const val RIGHT_SHOULDER = 12
    const val LEFT_ELBOW = 13
    const val RIGHT_ELBOW = 14
    const val LEFT_WRIST = 15
    const val RIGHT_WRIST = 16
    const val LEFT_HIP = 23
    const val RIGHT_HIP = 24
    const val LEFT_KNEE = 25
    const val RIGHT_KNEE = 26
    const val LEFT_ANKLE = 27
    const val RIGHT_ANKLE = 28
    const val LEFT_HEEL = 29
    const val RIGHT_HEEL = 30
    const val LEFT_FOOT = 31
    const val RIGHT_FOOT = 32
}

/** Lines of the body skeleton (face details left out). */
val BODY_CONNECTIONS: List<Pair<Int, Int>> = with(PoseIds) {
    listOf(
        LEFT_SHOULDER to RIGHT_SHOULDER,
        LEFT_SHOULDER to LEFT_ELBOW, LEFT_ELBOW to LEFT_WRIST,
        RIGHT_SHOULDER to RIGHT_ELBOW, RIGHT_ELBOW to RIGHT_WRIST,
        LEFT_SHOULDER to LEFT_HIP, RIGHT_SHOULDER to RIGHT_HIP, LEFT_HIP to RIGHT_HIP,
        LEFT_HIP to LEFT_KNEE, LEFT_KNEE to LEFT_ANKLE,
        RIGHT_HIP to RIGHT_KNEE, RIGHT_KNEE to RIGHT_ANKLE,
        LEFT_ANKLE to LEFT_HEEL, LEFT_HEEL to LEFT_FOOT, LEFT_ANKLE to LEFT_FOOT,
        RIGHT_ANKLE to RIGHT_HEEL, RIGHT_HEEL to RIGHT_FOOT, RIGHT_ANKLE to RIGHT_FOOT,
    )
}
