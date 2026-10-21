package dev.wardrobe.app.pipeline.pose

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Smoke test for the bundled pose model, run on the tablet.
 *
 * The second test needs a photo of me at app/src/androidTest/assets/pose_test.jpg (or .png). It is git-ignored
 * (never commit photos), so the test is skipped when the file is missing.
 */
@RunWith(AndroidJUnit4::class)
class PoseDetectorTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun modelLoadsAndRunsOnBlankImage() {
        val blank = Bitmap.createBitmap(720, 1280, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        PoseDetector(appContext).use { detector ->
            assertTrue(detector.detect(blank).landmarks().isEmpty())
        }
    }

    @Test
    fun findsMyPoseOnTestPhoto() {
        val photoName = TEST_PHOTOS.firstOrNull { testAssets.list("")?.contains(it) == true }
        assumeTrue("Put a full-body photo at androidTest/assets/pose_test.jpg (or .png) to run this", photoName != null)
        val photo = testAssets.open(photoName!!).use { BitmapFactory.decodeStream(it) }

        PoseDetector(appContext).use { detector ->
            detector.detect(photo) // warm-up
            val start = System.nanoTime()
            val result = detector.detect(photo)
            val ms = (System.nanoTime() - start) / 1_000_000

            assertEquals("people found", 1, result.landmarks().size)
            val landmarks = result.landmarks()[0]
            assertEquals(33, landmarks.size)
            Log.i(TAG, "Pose on ${photo.width}x${photo.height} in $ms ms")
            landmarks.forEachIndexed { i, p ->
                Log.i(TAG, "landmark %2d  x=%.3f y=%.3f visibility=%.2f".format(i, p.x(), p.y(), p.visibility().orElse(0f)))
            }
        }
    }

    private companion object {
        val TEST_PHOTOS = listOf("pose_test.jpg", "pose_test.png")
        const val TAG = "PoseDetectorTest"
    }
}
