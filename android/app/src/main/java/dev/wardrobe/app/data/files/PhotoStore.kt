package dev.wardrobe.app.data.files

import dev.wardrobe.app.domain.model.View
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Owns every image file the app keeps. All files live under [root] (the app's private filesDir in
 * production), never in MediaStore / Gallery / shared storage. The database stores the relative
 * paths returned here, so the folder can be moved or backed up as a whole.
 *
 * Layout:
 *   body/<figureKey>/<view>.jpg                 body figure photos
 *   captures/<sessionId>/<garment>/<view>.jpg   raw captures, garment = 1, 2, … within the session (+ closeup.jpg)
 *   cutouts/<entryId>/<view>.png                batch-processing output awaiting review
 *   items/<itemId>/<versionId>/<view>.png       approved wardrobe images
 */
class PhotoStore(private val root: File) {

    fun bodyPath(figureKey: String, view: View) = "body/$figureKey/${view.key}.jpg"
    fun rawCapturePath(sessionId: String, garment: Int, view: View) = "captures/$sessionId/$garment/${view.key}.jpg"
    fun closeUpPath(sessionId: String, garment: Int) = "captures/$sessionId/$garment/closeup.jpg"
    fun captureDir(sessionId: String, garment: Int) = "captures/$sessionId/$garment"
    fun cutoutPath(entryId: Long, view: View) = "cutouts/$entryId/${view.key}.png"
    fun itemPath(itemId: Long, versionId: Long, view: View) = "items/$itemId/$versionId/${view.key}.png"
    /** Manual shutter shots outside a session (Phase 1 testing). */

    /** Absolute file for a stored relative path. Rejects paths that would escape [root]. */
    fun file(relativePath: String): File {
        val base = root.canonicalFile
        val target = File(base, relativePath).canonicalFile
        require(target.path.startsWith(base.path + File.separator)) { "Path escapes photo store: $relativePath" }
        return target
    }

    /**
     * Writes [bytes] so that the file is either fully written or absent: write to a temp file,
     * flush to disk, then rename. Returns [relativePath] for storing in the database.
     */
    fun write(relativePath: String, bytes: ByteArray): String {
        val target = file(relativePath)
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, target.name + ".tmp")
        FileOutputStream(temp).use { out ->
            out.write(bytes)
            out.fd.sync()
        }
        // Replaces an existing file (e.g. a redone shot) in one step.
        Files.move(temp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
        return relativePath
    }

    fun exists(relativePath: String): Boolean = file(relativePath).isFile

    fun delete(relativePath: String): Boolean = file(relativePath).delete()

    /** Deletes a whole folder, e.g. "captures/<sessionId>/<entryId>" after review. */
    fun deleteDir(relativeDir: String): Boolean = file(relativeDir).deleteRecursively()
}
