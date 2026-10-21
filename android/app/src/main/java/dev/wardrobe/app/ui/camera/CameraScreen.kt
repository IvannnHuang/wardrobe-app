package dev.wardrobe.app.ui.camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.ui.draw.clip
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import java.io.ByteArrayOutputStream
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.SystemClock
import dev.wardrobe.app.capture.LivePoseAnalyzer
import dev.wardrobe.app.domain.capture.AutoShutter
import dev.wardrobe.app.domain.capture.PoseCheckResult
import dev.wardrobe.app.domain.capture.PoseChecker
import dev.wardrobe.app.domain.capture.spoken
import dev.wardrobe.app.domain.model.View
import dev.wardrobe.app.domain.model.PoseFrame
import java.util.concurrent.Executors

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    var asked by remember { mutableStateOf(false) }
    val requestPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        granted = it
        asked = true
    }
    LaunchedEffect(Unit) {
        if (!granted) requestPermission.launch(Manifest.permission.CAMERA)
    }

    if (granted) {
        CameraControls()
    } else {
        CameraPermissionNeeded(
            denied = asked,
            onAllow = { requestPermission.launch(Manifest.permission.CAMERA) },
            onOpenSettings = {
                context.startActivity(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
                )
            },
        )
    }
}

@Composable
private fun CameraControls(viewModel: CameraViewModel = hiltViewModel()) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsStateWithLifecycle()
    val imageCapture = remember { buildImageCapture() }
    val imageAnalysis = remember { LivePoseAnalyzer.buildImageAnalysis() }
    val snackbar = remember { SnackbarHostState() }

    // Live pose: the model runs on its own thread; the latest result is drawn over the preview.
    var pose by remember { mutableStateOf<PoseFrame?>(null) }
    DisposableEffect(imageAnalysis) {
        val executor = Executors.newSingleThreadExecutor()
        val analyzer = LivePoseAnalyzer(context) { pose = it }
        imageAnalysis.setAnalyzer(executor, analyzer)
        onDispose {
            imageAnalysis.clearAnalyzer()
            executor.execute { analyzer.close() } // after any frame still being analysed
            executor.shutdown()
        }
    }

    // Pose checks + auto-capture
    val checker = remember { PoseChecker() }
    val autoShutter = remember { AutoShutter() }
    var wanted by remember { mutableStateOf(View.FRONT) }
    var autoMode by remember { mutableStateOf(false) }
    var check by remember { mutableStateOf<PoseCheckResult?>(null) }
    val beeper = remember { ToneGenerator(AudioManager.STREAM_MUSIC, 90) }
    DisposableEffect(Unit) { onDispose { beeper.release() } }

    LaunchedEffect(state.error) {
        state.error?.let {
            snackbar.showSnackbar(it)
            viewModel.errorShown()
        }
    }

    fun takePhoto() {
        viewModel.onCaptureStarted()
        // Write into memory first so CameraX includes the EXIF rotation; the ViewModel then saves it.
        val buffer = ByteArrayOutputStream()
        imageCapture.takePicture(
            ImageCapture.OutputFileOptions.Builder(buffer).build(),
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) =
                    viewModel.savePhoto(buffer.toByteArray())

                override fun onError(e: ImageCaptureException) =
                    viewModel.onCaptureFailed("Capture failed: ${e.message}")
            },
        )
    }

    LaunchedEffect(pose) {
        val frame = pose ?: return@LaunchedEffect
        val result = checker.check(frame, wanted, SystemClock.uptimeMillis())
        check = result
        if (autoMode && !state.capturing && autoShutter.update(result.ready, SystemClock.uptimeMillis())) {
            beeper.startTone(ToneGenerator.TONE_PROP_BEEP, 150)
            takePhoto()
            // Next angle: front -> right -> back -> left, then stop.
            val next = View.entries.getOrNull(wanted.ordinal + 1)
            if (next != null) wanted = next else autoMode = false
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CameraPreview(imageCapture, imageAnalysis, Modifier.fillMaxSize())
        PoseOverlay(
            pose, Modifier.fillMaxSize(),
            color = if (check?.ready == true) Color(0xFF4CFF7A) else Color(0xFFFFB74D),
        )
        PoseReadout(pose, check, Modifier.align(Alignment.TopStart).padding(16.dp))
        HintBanner(wanted, check, autoMode, Modifier.align(Alignment.TopCenter).padding(top = 16.dp))

        CaptureSettings(
            wanted = wanted,
            onWanted = { wanted = it },
            autoMode = autoMode,
            onAutoMode = { autoMode = it },
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
        )

        state.lastShot?.let { shot ->
            LastShotCard(shot, state.shotsSaved, Modifier.align(Alignment.BottomStart).padding(16.dp))
        }

        ShutterButton(
            busy = state.capturing,
            onClick = ::takePhoto,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 24.dp),
        )

        SnackbarHost(snackbar, Modifier.align(Alignment.TopCenter))
    }
}

@Composable
private fun HintBanner(wanted: View, check: PoseCheckResult?, autoMode: Boolean, modifier: Modifier = Modifier) {
    val ready = check?.ready == true
    val text = when {
        check == null -> "Starting…"
        ready -> "${wanted.spoken().replaceFirstChar { it.uppercase() }} ✓ Ready" + if (autoMode) " · capturing" else ""
        else -> check.issues.first().message
    }
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(24.dp),
        color = if (ready) Color(0xFF1B5E20).copy(alpha = 0.85f) else Color.Black.copy(alpha = 0.7f),
    ) {
        Text(text, Modifier.padding(horizontal = 24.dp, vertical = 12.dp), color = Color.White,
            style = MaterialTheme.typography.headlineSmall)
    }
}

@Composable
private fun CaptureSettings(
    wanted: View,
    onWanted: (View) -> Unit,
    autoMode: Boolean,
    onAutoMode: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(modifier = modifier, shape = RoundedCornerShape(12.dp), color = Color.Black.copy(alpha = 0.6f)) {
        Column(Modifier.padding(12.dp), horizontalAlignment = Alignment.End) {
            Row {
                View.entries.forEach { view ->
                    FilterChip(
                        selected = view == wanted,
                        onClick = { onWanted(view) },
                        label = { Text(view.spoken().substringBefore(" ").replaceFirstChar { it.uppercase() }) },
                        modifier = Modifier.padding(start = 6.dp),
                        colors = FilterChipDefaults.filterChipColors(labelColor = Color.White),
                    )
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Auto capture", color = Color.White, style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.width(8.dp))
                Switch(checked = autoMode, onCheckedChange = onAutoMode)
            }
        }
    }
}

@Composable
private fun PoseReadout(pose: PoseFrame?, check: PoseCheckResult?, modifier: Modifier = Modifier) {
    val text = when {
        pose == null -> "Pose: starting…"
        pose.landmarks.isEmpty() -> "Pose: nobody in view · ${pose.inferenceMs} ms"
        else -> "Pose: ${pose.inferenceMs} ms · seen ${check?.seenView?.spoken() ?: "?"} · " +
            "shoulders/torso %.2f · face %.2f · L-shoulder on image %s".format(
                check?.shoulderRatio ?: 0f, check?.faceVisibility ?: 0f,
                if (check?.leftShoulderOnImageRight == true) "right" else "left",
            )
    }
    Surface(modifier = modifier, shape = RoundedCornerShape(8.dp), color = Color.Black.copy(alpha = 0.6f)) {
        Text(text, Modifier.padding(horizontal = 10.dp, vertical = 6.dp), color = Color.White,
            style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ShutterButton(busy: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .size(84.dp)
            .border(4.dp, Color.White, CircleShape)
            .padding(8.dp)
            .clip(CircleShape)
            .background(if (busy) Color.Gray else Color.White)
            .clickable(enabled = !busy, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        if (busy) CircularProgressIndicator(Modifier.size(32.dp), color = Color.White, strokeWidth = 3.dp)
    }
}

@Composable
private fun LastShotCard(shot: CameraViewModel.LastShot, saved: Int, modifier: Modifier = Modifier) {
    Surface(modifier = modifier, shape = RoundedCornerShape(12.dp), color = Color.Black.copy(alpha = 0.6f)) {
        Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Image(
                bitmap = shot.thumbnail,
                contentDescription = "Last photo",
                modifier = Modifier.height(96.dp).clip(RoundedCornerShape(8.dp)),
            )
            Spacer(Modifier.width(12.dp))
            Column {
                Text("${shot.width} × ${shot.height}", color = Color.White, style = MaterialTheme.typography.titleSmall)
                Text(
                    "%.1f MP · %d saved".format(shot.width * shot.height / 1e6, saved),
                    color = Color.White.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun CameraPermissionNeeded(denied: Boolean, onAllow: () -> Unit, onOpenSettings: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Camera access needed", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(12.dp))
        Text(
            "The Camera page photographs each garment from 4 angles. Photos stay in the app's private " +
                "storage on this tablet and are never uploaded.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onAllow) { Text("Allow camera") }
        if (denied) {
            // After "Don't allow" twice, Android stops showing the prompt; only Settings can grant it.
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = onOpenSettings) { Text("Open app settings") }
        }
    }
}
