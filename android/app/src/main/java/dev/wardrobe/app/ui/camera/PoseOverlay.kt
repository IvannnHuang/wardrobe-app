package dev.wardrobe.app.ui.camera

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.unit.dp
import dev.wardrobe.app.domain.model.BODY_CONNECTIONS
import dev.wardrobe.app.domain.model.PoseFrame
import kotlin.math.min

private const val MIN_VISIBILITY = 0.5f

/**
 * Draws the detected skeleton over the camera preview. The preview is shown "fit center", so the
 * camera image occupies a centred rectangle of the same aspect ratio; landmarks are mapped into it.
 */
@Composable
fun PoseOverlay(pose: PoseFrame?, modifier: Modifier = Modifier, color: Color = Color(0xFF4CFF7A)) {
    Canvas(modifier) {
        if (pose == null || pose.landmarks.isEmpty()) return@Canvas
        val scale = min(size.width / pose.imageWidth, size.height / pose.imageHeight)
        val left = (size.width - pose.imageWidth * scale) / 2
        val top = (size.height - pose.imageHeight * scale) / 2
        fun at(i: Int) = pose.landmarks[i].let {
            Offset(left + it.x * pose.imageWidth * scale, top + it.y * pose.imageHeight * scale)
        }
        fun visible(i: Int) = pose.landmarks[i].visibility >= MIN_VISIBILITY

        val stroke = 5.dp.toPx()
        for ((a, b) in BODY_CONNECTIONS) {
            if (visible(a) && visible(b)) drawLine(color, at(a), at(b), stroke, StrokeCap.Round)
        }
        for (i in 11 until pose.landmarks.size) {
            if (visible(i)) drawCircle(Color.White, radius = stroke, center = at(i))
        }
        if (visible(0)) drawCircle(color, radius = stroke * 2.5f, center = at(0))
    }
}
