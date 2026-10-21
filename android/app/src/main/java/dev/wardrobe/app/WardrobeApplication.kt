package dev.wardrobe.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/** App entry point. @HiltAndroidApp sets up dependency injection for the whole app. */
@HiltAndroidApp
class WardrobeApplication : Application()
