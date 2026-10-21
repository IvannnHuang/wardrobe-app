package dev.wardrobe.app.pipeline

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import dagger.hilt.android.qualifiers.ApplicationContext
import dev.wardrobe.app.data.db.CaptureEntryDao
import dev.wardrobe.app.data.db.CaptureStatus
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/** Queues batch processing. One unique job per garment, run one at a time to keep the tablet responsive. */
@Singleton
class ProcessingScheduler @Inject constructor(
    @ApplicationContext private val context: Context,
    private val captureEntryDao: CaptureEntryDao,
) {
    fun enqueue(entryIds: List<Long>) {
        val wm = WorkManager.getInstance(context)
        entryIds.forEach { id ->
            val request = OneTimeWorkRequestBuilder<ProcessCaptureWorker>()
                .setInputData(workDataOf(ProcessCaptureWorker.KEY_ENTRY_ID to id))
                .setBackoffCriteria(BackoffPolicy.LINEAR, 30, TimeUnit.SECONDS)
                .addTag(TAG)
                .build()
            // One appending queue, so garments are processed one after another. A garment queued twice
            // is harmless: the worker skips entries that are no longer waiting.
            wm.enqueueUniqueWork(QUEUE, ExistingWorkPolicy.APPEND_OR_REPLACE, request)
        }
    }

    /** Anything captured but not processed yet, e.g. after the app was closed before Finish. */
    suspend fun enqueuePending() =
        enqueue(captureEntryDao.idsWithStatus(listOf(CaptureStatus.CAPTURED, CaptureStatus.PROCESSING)))

    private companion object {
        const val QUEUE = "process-capture"
        const val TAG = "processing"
    }
}
