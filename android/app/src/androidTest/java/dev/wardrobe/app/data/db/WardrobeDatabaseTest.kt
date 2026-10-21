package dev.wardrobe.app.data.db

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** Runs on the tablet against a throwaway in-memory database. */
@RunWith(AndroidJUnit4::class)
class WardrobeDatabaseTest {
    private lateinit var db: WardrobeDatabase

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        db = Room.inMemoryDatabaseBuilder(context, WardrobeDatabase::class.java).build()
    }

    @After
    fun tearDown() = db.close()

    private suspend fun newItem(): Long = db.itemDao().insertItem(
        ItemEntity(name = "White tee", category = "t-shirt", rank = "U1", defaultVersionId = null, createdAt = 0)
    )

    private fun modelTag(itemId: Long, key: String, value: String) =
        TagEntity(itemId, key, value, confidence = 0.6f, source = TagSource.MODEL, updatedAt = 1)

    @Test
    fun rerunningTaggingNeverOverwritesUserTags() = runBlocking {
        val itemId = newItem()
        val tags = db.tagDao()

        // Model suggests a color, I correct it.
        tags.putModelTag(modelTag(itemId, "color", "\"cream\""))
        tags.upsertUserTag(TagEntity(itemId, "color", "\"white\"", 1f, TagSource.USER, updatedAt = 2))

        // Re-run tagging: color must stay mine; a new key is added; a model-only key is refreshed.
        tags.putModelTag(modelTag(itemId, "sleeve", "\"short\""))
        tags.putModelTag(modelTag(itemId, "color", "\"beige\""))
        tags.putModelTag(modelTag(itemId, "sleeve", "\"cap\""))

        val color = tags.getTag(itemId, "color")!!
        assertEquals("\"white\"", color.valueJson)
        assertEquals(TagSource.USER, color.source)
        assertEquals("\"cap\"", tags.getTag(itemId, "sleeve")!!.valueJson)
    }

    @Test
    fun deletingItemRemovesItsVersionsAndTags() = runBlocking {
        val itemId = newItem()
        db.itemDao().insertVersion(
            VersionEntity(
                itemId = itemId, label = "default", frontPath = "f", rightPath = "r", backPath = "b",
                leftPath = "l", closeUpPath = null, landmarksJson = "{}", capturedAt = 0,
            )
        )
        db.tagDao().putModelTag(modelTag(itemId, "color", "\"white\""))

        db.itemDao().deleteItem(db.itemDao().getItem(itemId)!!)

        assertTrue(db.itemDao().observeVersions(itemId).first().isEmpty())
        assertNull(db.tagDao().getTag(itemId, "color"))
    }
}
