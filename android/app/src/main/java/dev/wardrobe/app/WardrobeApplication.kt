package dev.wardrobe.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import dev.wardrobe.app.pipeline.ProcessingScheduler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject

/** App entry point. @HiltAndroidApp sets up dependency injection for the whole app. */
@HiltAndroidApp
class WardrobeApplication : Application() {

    @Inject lateinit var processingScheduler: ProcessingScheduler

    override fun onCreate() {
        super.onCreate()
        // Resume processing of anything captured earlier but not processed yet.
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch { processingScheduler.enqueuePending() }
    }
}
