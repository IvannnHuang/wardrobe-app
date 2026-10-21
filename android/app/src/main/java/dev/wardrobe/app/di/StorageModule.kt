package dev.wardrobe.app.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import dev.wardrobe.app.data.files.PhotoStore
import java.io.File
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object StorageModule {

    /** Photos live in app-private storage only (filesDir/photos). */
    @Provides
    @Singleton
    fun providePhotoStore(@ApplicationContext context: Context): PhotoStore =
        PhotoStore(File(context.filesDir, "photos"))
}
