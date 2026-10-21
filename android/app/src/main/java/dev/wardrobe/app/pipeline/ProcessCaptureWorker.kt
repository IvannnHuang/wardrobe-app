package dev.wardrobe.app.pipeline

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import dagger.hilt.EntryPoint
import dagger.hilt.InstallIn
import dagger.hilt.android.EntryPointAccessors
import dagger.hilt.components.SingletonComponent
import dev.wardrobe.app.data.db.CaptureEntryDao
import dev.wardrobe.app.data.db.CaptureStatus

/**
 * Processes one review-tray garment after the session: CAPTURED → PROCESSING → READY_FOR_REVIEW.
 * WorkManager keeps the job across app restarts and reboots; if the app dies mid-way the job runs
 * again from the start (each step overwrites its own output, so re-running is safe).
 */
class ProcessCaptureWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    @EntryPoint
    @InstallIn(SingletonComponent::class)
    interface Deps {
        fun captureEntryDao(): CaptureEntryDao
        fun captureProcessor(): CaptureProcessor
        fun readyNotifier(): ReadyNotifier
    }

    private val deps = EntryPointAccessors.fromApplication(context.applicationContext, Deps::class.java)

    override suspend fun doWork(): Result {
        val dao = deps.captureEntryDao()
        val entryId = inputData.getLong(KEY_ENTRY_ID, -1)
        val entry = dao.get(entryId) ?: return Result.success() // deleted meanwhile
        if (entry.status !in setOf(CaptureStatus.CAPTURED, CaptureStatus.PROCESSING)) return Result.success()

        dao.update(entry.copy(status = CaptureStatus.PROCESSING))
        return try {
            val output = deps.captureProcessor().process(entry)
            dao.update(entry.copy(status = CaptureStatus.READY_FOR_REVIEW, cutoutPathsJson = output.cutoutPathsJson))
            deps.readyNotifier().notifyIfBatchDone()
            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Processing entry $entryId failed (attempt ${runAttemptCount + 1})", e)
            if (runAttemptCount + 1 < MAX_ATTEMPTS) {
                Result.retry()
            } else {
                dao.update(entry.copy(status = CaptureStatus.FAILED))
                deps.readyNotifier().notifyIfBatchDone()
                // Not Result.failure(): that would cancel the garments queued after this one.
                Result.success()
            }
        }
    }

    companion object {
        const val KEY_ENTRY_ID = "entryId"
        private const val MAX_ATTEMPTS = 3
        private const val TAG = "ProcessCaptureWorker"
    }
}
