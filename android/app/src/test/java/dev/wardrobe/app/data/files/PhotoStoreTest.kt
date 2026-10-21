package dev.wardrobe.app.data.files

import dev.wardrobe.app.domain.model.View
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** Plain JVM test: runs on the computer, no tablet needed. */
class PhotoStoreTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val store by lazy { PhotoStore(tmp.root) }

    @Test
    fun writeThenReadBack() {
        val path = store.write(store.rawCapturePath("s1", 7, View.FRONT), byteArrayOf(1, 2, 3))

        assertEquals("captures/s1/7/front.jpg", path)
        assertTrue(store.exists(path))
        assertArrayEquals(byteArrayOf(1, 2, 3), store.file(path).readBytes())
        assertFalse(store.file("$path.tmp").exists())
    }

    @Test
    fun redoReplacesExistingPhoto() {
        val path = store.rawCapturePath("s1", 7, View.BACK)
        store.write(path, byteArrayOf(1))
        store.write(path, byteArrayOf(2))

        assertArrayEquals(byteArrayOf(2), store.file(path).readBytes())
    }

    @Test
    fun deleteDirRemovesEverything() {
        View.entries.forEach { store.write(store.rawCapturePath("s1", 7, it), byteArrayOf(0)) }

        assertTrue(store.deleteDir("captures/s1/7"))
        View.entries.forEach { assertFalse(store.exists(store.rawCapturePath("s1", 7, it))) }
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsPathsOutsideStore() {
        store.write("../outside.jpg", byteArrayOf(0))
    }
}
