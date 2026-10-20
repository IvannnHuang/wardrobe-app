package dev.wardrobe.app.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Checkroom
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Style
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import dev.wardrobe.app.ui.camera.CameraScreen
import dev.wardrobe.app.ui.pairing.PairingScreen
import dev.wardrobe.app.ui.wardrobe.WardrobeScreen

enum class Page(val route: String, val label: String, val icon: ImageVector) {
    Pairing("pairing", "Pairing", Icons.Filled.Style),
    Wardrobe("wardrobe", "Wardrobe", Icons.Filled.Checkroom),
    Camera("camera", "Camera", Icons.Filled.PhotoCamera),
}

@Composable
fun WardrobeApp() {
    val navController = rememberNavController()
    val backStack by navController.currentBackStackEntryAsState()
    val currentDestination = backStack?.destination

    Scaffold(
        bottomBar = {
            NavigationBar {
                Page.entries.forEach { page ->
                    val selected = currentDestination?.hierarchy?.any { it.route == page.route } == true
                    NavigationBarItem(
                        selected = selected,
                        onClick = {
                            navController.navigate(page.route) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(page.icon, contentDescription = page.label) },
                        label = { Text(page.label) },
                    )
                }
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Page.Pairing.route,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Page.Pairing.route) { PairingScreen() }
            composable(Page.Wardrobe.route) { WardrobeScreen() }
            composable(Page.Camera.route) { CameraScreen() }
        }
    }
}
