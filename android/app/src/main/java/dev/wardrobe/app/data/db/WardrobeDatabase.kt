package dev.wardrobe.app.data.db

import androidx.room.Database
import androidx.room.RoomDatabase

/**
 * The app's SQLite database (stored in app-private storage).
 * Bump [version] and add a migration whenever an entity changes, so my wardrobe data is kept.
 */
@Database(
    entities = [
        BodyFigureEntity::class,
        ItemEntity::class,
        VersionEntity::class,
        TagEntity::class,
        CaptureEntryEntity::class,
        OutfitEntity::class,
        OutfitItemEntity::class,
        WearLogEntity::class,
        TagCorrectionEntity::class,
        LayerRuleEntity::class,
    ],
    version = 1,
    exportSchema = true,
)
abstract class WardrobeDatabase : RoomDatabase() {
    abstract fun bodyFigureDao(): BodyFigureDao
    abstract fun itemDao(): ItemDao
    abstract fun tagDao(): TagDao
    abstract fun captureEntryDao(): CaptureEntryDao
    abstract fun outfitDao(): OutfitDao
    abstract fun layerRuleDao(): LayerRuleDao
}
