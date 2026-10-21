package dev.wardrobe.app.di

import android.content.Context
import androidx.room.Room
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.wardrobe.app.data.db.WardrobeDatabase
import javax.inject.Singleton

/** Tells Hilt how to build the database (one instance for the whole app) and its DAOs. */
@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): WardrobeDatabase =
        Room.databaseBuilder(context, WardrobeDatabase::class.java, "wardrobe.db").build()

    @Provides fun bodyFigureDao(db: WardrobeDatabase) = db.bodyFigureDao()
    @Provides fun itemDao(db: WardrobeDatabase) = db.itemDao()
    @Provides fun tagDao(db: WardrobeDatabase) = db.tagDao()
    @Provides fun captureEntryDao(db: WardrobeDatabase) = db.captureEntryDao()
    @Provides fun outfitDao(db: WardrobeDatabase) = db.outfitDao()
    @Provides fun layerRuleDao(db: WardrobeDatabase) = db.layerRuleDao()
}
