package dev.wardrobe.app.data.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

// Room tables, following the data model in CLAUDE.md.
// Conventions:
//  - Image paths are relative to the app's filesDir (see the file store), never shared storage.
//  - Landmarks, tag values and other structured data are JSON strings; parsing lives in domain/.
//  - Times are epoch milliseconds.

/** My body in the base outfit: 4 views + pose landmarks, used to align every garment. */
@Entity(tableName = "body_figure")
data class BodyFigureEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val frontPath: String,
    val rightPath: String,
    val backPath: String,
    val leftPath: String,
    /** JSON: landmarks per view, keyed by "front" | "right" | "back" | "left". */
    val landmarksJson: String,
    val heightCm: Float,
    val createdAt: Long,
)

enum class Availability { AVAILABLE, LAUNDRY, LENT, STORED }

/** One physical garment. Its photos live in [VersionEntity]. */
@Entity(tableName = "item", indices = [Index("category"), Index("rank")])
data class ItemEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val category: String,
    /** Layer rank code from CLAUDE.md: U1..U5, L1, L2, F, H. */
    val rank: String,
    /** No foreign key: Item and Version point at each other. */
    val defaultVersionId: Long?,
    val availability: Availability = Availability.AVAILABLE,
    val createdAt: Long,
)

/** One way of wearing an item (e.g. "tucked", "zipped", "open"), with its 4 cut-out views. */
@Entity(
    tableName = "version",
    foreignKeys = [ForeignKey(ItemEntity::class, ["id"], ["itemId"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("itemId")],
)
data class VersionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val itemId: Long,
    val label: String,
    val frontPath: String,
    val rightPath: String,
    val backPath: String,
    val leftPath: String,
    val closeUpPath: String?,
    val landmarksJson: String,
    val capturedAt: Long,
)

enum class TagSource { USER, MODEL }

/** One tag per (item, key). A USER tag is my verified value and must never be overwritten by the model. */
@Entity(
    tableName = "tag",
    primaryKeys = ["itemId", "key"],
    foreignKeys = [ForeignKey(ItemEntity::class, ["id"], ["itemId"], onDelete = ForeignKey.CASCADE)],
)
data class TagEntity(
    val itemId: Long,
    val key: String,
    val valueJson: String,
    val confidence: Float,
    val source: TagSource,
    val updatedAt: Long,
)

enum class CaptureStatus { CAPTURED, PROCESSING, READY_FOR_REVIEW, APPROVED, REJECTED, FAILED }

/** One garment from a capture session, waiting in the review tray. Nothing here is in the wardrobe yet. */
@Entity(tableName = "capture_entry", indices = [Index("sessionId"), Index("status")])
data class CaptureEntryEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    /** JSON: raw photo paths keyed by view, plus optional "closeUp". */
    val rawPathsJson: String,
    val status: CaptureStatus = CaptureStatus.CAPTURED,
    val cutoutPathsJson: String? = null,
    val suggestedTagsJson: String? = null,
    val suggestedMatchesJson: String? = null,
    val createdAt: Long,
)

@Entity(tableName = "outfit")
data class OutfitEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val createdAt: Long,
)

@Entity(
    tableName = "outfit_item",
    primaryKeys = ["outfitId", "itemId"],
    foreignKeys = [
        ForeignKey(OutfitEntity::class, ["id"], ["outfitId"], onDelete = ForeignKey.CASCADE),
        ForeignKey(ItemEntity::class, ["id"], ["itemId"], onDelete = ForeignKey.CASCADE),
        ForeignKey(VersionEntity::class, ["id"], ["versionId"], onDelete = ForeignKey.CASCADE),
    ],
    indices = [Index("itemId"), Index("versionId")],
)
data class OutfitItemEntity(
    val outfitId: Long,
    val itemId: Long,
    val versionId: Long,
    /** JSON: per-outfit state such as tucked, zipped, bootsOverPants. */
    val stateFlagsJson: String = "{}",
)

@Entity(
    tableName = "wear_log",
    primaryKeys = ["outfitId", "epochDay"],
    foreignKeys = [ForeignKey(OutfitEntity::class, ["id"], ["outfitId"], onDelete = ForeignKey.CASCADE)],
)
data class WearLogEntity(
    val outfitId: Long,
    /** Day worn, as java.time.LocalDate.toEpochDay(). */
    val epochDay: Long,
    val comfortRating: Int? = null,
)

/** History of my tag edits, kept for learning from corrections later. */
@Entity(
    tableName = "tag_correction",
    foreignKeys = [ForeignKey(ItemEntity::class, ["id"], ["itemId"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("itemId")],
)
data class TagCorrectionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val itemId: Long,
    val key: String,
    val oldValueJson: String?,
    val newValueJson: String,
    val at: Long,
)

/** Editable layering config: which rank a category goes to and its rule flags. */
@Entity(tableName = "layer_rule")
data class LayerRuleEntity(
    @PrimaryKey val category: String,
    val rank: String,
    /** JSON: e.g. {"tuckable": true, "fillsRanks": ["U2","L2"], "canGoOverPants": false}. */
    val flagsJson: String = "{}",
)
