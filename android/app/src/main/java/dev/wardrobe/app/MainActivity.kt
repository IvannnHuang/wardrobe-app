package dev.wardrobe.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import dev.wardrobe.app.ui.navigation.WardrobeApp
import dev.wardrobe.app.ui.theme.WardrobeTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            WardrobeTheme {
                WardrobeApp()
            }
        }
    }
}
