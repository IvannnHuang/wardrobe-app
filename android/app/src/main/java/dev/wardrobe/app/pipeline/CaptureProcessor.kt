package dev.wardrobe.app.pipeline

import dev.wardrobe.app.data.db.CaptureEntryEntity
import javax.inject.Inject
import javax.inject.Singleton

/** Runs the processing steps for one garment. Cut-out and tagging are filled in by later steps. */
@Singleton
class CaptureProcessor @Inject constructor() {

    data class Output(val cutoutPathsJson: String?)

    suspend fun process(entry: CaptureEntryEntity): Output = Output(cutoutPathsJson = null)
}
