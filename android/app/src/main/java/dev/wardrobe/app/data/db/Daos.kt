package dev.wardrobe.app.data.db

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Update
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

// Basic queries per area. Flow-returning queries update the UI automatically when data changes.

@Dao
interface BodyFigureDao {
    @Insert
    suspend fun insert(figure: BodyFigureEntity): Long

    /** The most recent body figure is the active one. */
    @Query("SELECT * FROM body_figure ORDER BY createdAt DESC LIMIT 1")
    fun observeCurrent(): Flow<BodyFigureEntity?>
}

@Dao
interface ItemDao {
    @Insert
    suspend fun insertItem(item: ItemEntity): Long

    @Update
    suspend fun updateItem(item: ItemEntity)

    @Delete
    suspend fun deleteItem(item: ItemEntity)

    @Query("SELECT * FROM item WHERE id = :id")
    suspend fun getItem(id: Long): ItemEntity?

    @Query("SELECT * FROM item ORDER BY category, name")
    fun observeItems(): Flow<List<ItemEntity>>

    @Insert
    suspend fun insertVersion(version: VersionEntity): Long

    @Query("SELECT * FROM version WHERE itemId = :itemId ORDER BY capturedAt")
    fun observeVersions(itemId: Long): Flow<List<VersionEntity>>
}

@Dao
interface TagDao {
    @Query("SELECT * FROM tag WHERE itemId = :itemId")
    fun observeTags(itemId: Long): Flow<List<TagEntity>>

    @Query("SELECT * FROM tag WHERE itemId = :itemId AND `key` = :key")
    suspend fun getTag(itemId: Long, key: String): TagEntity?

    /** My edit or approval: always wins. */
    @Upsert
    suspend fun upsertUserTag(tag: TagEntity)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertIfAbsent(tag: TagEntity): Long

    @Query(
        "UPDATE tag SET valueJson = :valueJson, confidence = :confidence, updatedAt = :updatedAt " +
            "WHERE itemId = :itemId AND `key` = :key AND source = 'MODEL'"
    )
    suspend fun updateModelTag(itemId: Long, key: String, valueJson: String, confidence: Float, updatedAt: Long)

    /** Re-running tagging goes through here: it adds new tags and refreshes MODEL tags, but never touches USER tags. */
    @Transaction
    suspend fun putModelTag(tag: TagEntity) {
        require(tag.source == TagSource.MODEL) { "putModelTag is only for model tags" }
        if (insertIfAbsent(tag) == -1L) {
            updateModelTag(tag.itemId, tag.key, tag.valueJson, tag.confidence, tag.updatedAt)
        }
    }

    @Insert
    suspend fun insertCorrection(correction: TagCorrectionEntity)
}

@Dao
interface CaptureEntryDao {
    @Insert
    suspend fun insert(entry: CaptureEntryEntity): Long

    @Update
    suspend fun update(entry: CaptureEntryEntity)

    @Query("SELECT * FROM capture_entry WHERE id = :id")
    suspend fun get(id: Long): CaptureEntryEntity?

    @Query("SELECT * FROM capture_entry WHERE status = :status ORDER BY createdAt")
    fun observeByStatus(status: CaptureStatus): Flow<List<CaptureEntryEntity>>

    @Query("SELECT id FROM capture_entry WHERE status IN (:statuses) ORDER BY createdAt")
    suspend fun idsWithStatus(statuses: List<CaptureStatus>): List<Long>

    /** Garments waiting in the review tray (captured but not yet approved or rejected). */
    @Query("SELECT COUNT(*) FROM capture_entry WHERE status IN ('CAPTURED', 'PROCESSING', 'READY_FOR_REVIEW')")
    fun observeTrayCount(): Flow<Int>
}

@Dao
interface OutfitDao {
    @Insert
    suspend fun insertOutfit(outfit: OutfitEntity): Long

    @Insert
    suspend fun insertOutfitItems(items: List<OutfitItemEntity>)

    @Query("SELECT * FROM outfit ORDER BY createdAt DESC")
    fun observeOutfits(): Flow<List<OutfitEntity>>

    @Query("SELECT * FROM outfit_item WHERE outfitId = :outfitId")
    suspend fun getOutfitItems(outfitId: Long): List<OutfitItemEntity>

    @Upsert
    suspend fun logWear(entry: WearLogEntity)
}

@Dao
interface LayerRuleDao {
    @Query("SELECT * FROM layer_rule")
    fun observeRules(): Flow<List<LayerRuleEntity>>

    @Upsert
    suspend fun upsert(rules: List<LayerRuleEntity>)
}
