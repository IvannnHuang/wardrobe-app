package dev.wardrobe.app.pipeline

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.wardrobe.app.MainActivity
import dev.wardrobe.app.data.db.CaptureEntryDao
import dev.wardrobe.app.data.db.CaptureStatus
import javax.inject.Inject
import javax.inject.Singleton

/** Posts "N items ready to review" once nothing is left to process. Local notification only. */
@Singleton
class ReadyNotifier @Inject constructor(
    @ApplicationContext private val context: Context,
    private val captureEntryDao: CaptureEntryDao,
) {
    suspend fun notifyIfBatchDone() {
        val pending = captureEntryDao.idsWithStatus(listOf(CaptureStatus.CAPTURED, CaptureStatus.PROCESSING))
        if (pending.isNotEmpty()) return
        val ready = captureEntryDao.idsWithStatus(listOf(CaptureStatus.READY_FOR_REVIEW)).size
        val failed = captureEntryDao.idsWithStatus(listOf(CaptureStatus.FAILED)).size
        if (ready == 0 && failed == 0) return
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) return

        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL, "Processing", NotificationManager.IMPORTANCE_DEFAULT).apply {
                description = "When captured garments are ready to review"
            }
        )
        val open = PendingIntent.getActivity(
            context, 0, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val text = buildString {
            append("$ready item${if (ready == 1) "" else "s"} ready to review")
            if (failed > 0) append(" · $failed failed")
        }
        val notification = NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setContentTitle("Wardrobe")
            .setContentText(text)
            .setContentIntent(open)
            .setAutoCancel(true)
            .build()
        @Suppress("MissingPermission") // checked above
        NotificationManagerCompat.from(context).notify(NOTIFICATION_ID, notification)
    }

    private companion object {
        const val CHANNEL = "processing"
        const val NOTIFICATION_ID = 1
    }
}
