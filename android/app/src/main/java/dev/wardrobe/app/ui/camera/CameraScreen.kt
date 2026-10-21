package dev.wardrobe.app.ui.camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.net.Uri
import android.os.SystemClock
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
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.wardrobe.app.capture.LivePoseAnalyzer
import dev.wardrobe.app.domain.capture.AutoShutter
import dev.wardrobe.app.domain.capture.CaptureMode
import dev.wardrobe.app.domain.capture.CaptureSession
import dev.wardrobe.app.domain.capture.PoseCheckResult
import dev.wardrobe.app.domain.capture.PoseChecker
import dev.wardrobe.app.domain.capture.parseHeightCm
import dev.wardrobe.app.domain.capture.spoken
import dev.wardrobe.app.domain.model.PoseFrame
import dev.wardrobe.app.domain.model.View
import java.io.ByteArrayOutputStream
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
    val session = state.session
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

    val checker = remember { PoseChecker() }
    val autoShutter = remember { AutoShutter() }
    var autoMode by remember { mutableStateOf(true) }
    var check by remember { mutableStateOf<PoseCheckResult?>(null) }
    var confirmFinish by remember { mutableStateOf(false) }
    val beeper = remember { ToneGenerator(AudioManager.STREAM_MUSIC, 90) }
    DisposableEffect(Unit) { onDispose { beeper.release() } }

    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            viewModel.messageShown()
        }
    }
    // A longer tone when the 4th angle is done.
    LaunchedEffect(session?.complete) {
        if (session?.complete == true) beeper.startTone(ToneGenerator.TONE_PROP_ACK, 300)
    }

    fun takePhoto(view: View) {
        val landmarks = pose?.landmarks.orEmpty()
        viewModel.onCaptureStarted()
        // Write into memory first so CameraX includes the EXIF rotation; the ViewModel then saves it.
        val buffer = ByteArrayOutputStream()
        imageCapture.takePicture(
            ImageCapture.OutputFileOptions.Builder(buffer).build(),
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) =
                    viewModel.savePhoto(buffer.toByteArray(), view, landmarks)

                override fun onError(e: ImageCaptureException) =
                    viewModel.onCaptureFailed("Capture failed: ${e.message}")
            },
        )
    }

    LaunchedEffect(pose) {
        val frame = pose ?: return@LaunchedEffect
        val wanted = session?.wanted ?: View.FRONT
        val result = checker.check(frame, wanted, SystemClock.uptimeMillis())
        check = result
        val shoot = session?.wanted != null && autoMode && !state.capturing && !state.askHeight
        if (autoShutter.update(shoot && result.ready, SystemClock.uptimeMillis())) {
            beeper.startTone(ToneGenerator.TONE_PROP_BEEP, 150)
            takePhoto(wanted)
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CameraPreview(imageCapture, imageAnalysis, Modifier.fillMaxSize())
        PoseOverlay(
            pose, Modifier.fillMaxSize(),
            color = if (check?.ready == true) Color(0xFF4CFF7A) else Color(0xFFFFB74D),
        )
        PoseReadout(pose, check, Modifier.align(Alignment.TopStart).padding(16.dp))
        TrayBadge(state.trayCount, Modifier.align(Alignment.TopEnd).padding(16.dp))

        if (session == null) {
            StartPanel(
                hasBodyFigure = state.hasBodyFigure,
                onBodyFigure = { autoMode = true; viewModel.startBodyFigure() },
                onStartSession = { autoMode = true; viewModel.startSession() },
                modifier = Modifier.align(Alignment.Center),
            )
        } else {
            HintBanner(session, check, autoMode, Modifier.align(Alignment.TopCenter).padding(top = 64.dp))
            SessionPanel(
                session = session,
                saved = state.currentSaved,
                autoMode = autoMode,
                onAutoMode = { autoMode = it },
                onChoose = viewModel::choose,
                onNextGarment = viewModel::nextGarment,
                onFinish = {
                    if (session.mode == CaptureMode.GARMENT && session.started && !session.complete) confirmFinish = true
                    else viewModel.finish()
                },
                modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
            )
            session.wanted?.let { wanted ->
                ShutterButton(
                    busy = state.capturing,
                    onClick = { takePhoto(wanted) },
                    modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 24.dp),
                )
            }
        }

        state.lastShot?.let { shot ->
            LastShotCard(shot, Modifier.align(Alignment.BottomStart).padding(16.dp))
        }

        SnackbarHost(snackbar, Modifier.align(Alignment.Center))
    }

    if (state.askHeight) {
        HeightDialog(onSave = viewModel::saveBodyFigure, onCancel = viewModel::finish)
    }
    if (confirmFinish) {
        AlertDialog(
            onDismissRequest = { confirmFinish = false },
            title = { Text("Finish session?") },
            text = {
                Text("Garment ${session?.garmentNumber} has only ${session?.shots?.size} of 4 angles. Its photos will be deleted.")
            },
            confirmButton = { TextButton(onClick = { confirmFinish = false; viewModel.finish() }) { Text("Discard and finish") } },
            dismissButton = { TextButton(onClick = { confirmFinish = false }) { Text("Keep capturing") } },
        )
    }
}

@Composable
private fun StartPanel(
    hasBodyFigure: Boolean,
    onBodyFigure: () -> Unit,
    onStartSession: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(modifier = modifier, shape = RoundedCornerShape(16.dp), color = Color.Black.copy(alpha = 0.7f)) {
        Column(Modifier.padding(24.dp).width(360.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            if (hasBodyFigure) {
                Button(onClick = onStartSession, Modifier.fillMaxWidth()) { Text("Start session") }
                Spacer(Modifier.height(4.dp))
                Text(
                    "One garment at a time over the base outfit, 4 angles each.",
                    color = Color.White.copy(alpha = 0.8f), style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                )
                Spacer(Modifier.height(16.dp))
                OutlinedButton(onClick = onBodyFigure, Modifier.fillMaxWidth()) { Text("Redo body figure", color = Color.White) }
            } else {
                Text("First, set up your body figure", color = Color.White, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Wear only the base outfit. The app takes front, right, back and left, then asks your height.",
                    color = Color.White.copy(alpha = 0.8f), style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                )
                Spacer(Modifier.height(16.dp))
                Button(onClick = onBodyFigure, Modifier.fillMaxWidth()) { Text("Set up body figure") }
            }
        }
    }
}

@Composable
private fun HintBanner(session: CaptureSession, check: PoseCheckResult?, autoMode: Boolean, modifier: Modifier = Modifier) {
    val wanted = session.wanted
    val ready = wanted != null && check?.ready == true
    val text = when {
        wanted == null && session.mode == CaptureMode.BODY_FIGURE -> "All 4 angles done"
        wanted == null -> "Garment ${session.garmentNumber} done ✓ Change garment, then tap Next garment"
        check == null -> "Starting…"
        ready -> "${wanted.spoken().replaceFirstChar { it.uppercase() }} ✓ Ready" + if (autoMode) " · capturing" else ""
        else -> check.issues.first().message
    }
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(24.dp),
        color = if (ready || wanted == null) Color(0xFF1B5E20).copy(alpha = 0.85f) else Color.Black.copy(alpha = 0.7f),
    ) {
        Text(
            text, Modifier.padding(horizontal = 24.dp, vertical = 12.dp), color = Color.White,
            style = MaterialTheme.typography.headlineSmall,
        )
    }
}

@Composable
private fun SessionPanel(
    session: CaptureSession,
    saved: Boolean,
    autoMode: Boolean,
    onAutoMode: (Boolean) -> Unit,
    onChoose: (View) -> Unit,
    onNextGarment: () -> Unit,
    onFinish: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(modifier = modifier, shape = RoundedCornerShape(12.dp), color = Color.Black.copy(alpha = 0.7f)) {
        Column(Modifier.padding(12.dp), horizontalAlignment = Alignment.End) {
            val title = when (session.mode) {
                CaptureMode.BODY_FIGURE -> "Body figure"
                CaptureMode.GARMENT -> "Garment ${session.garmentNumber}" + if (saved) " · in review tray" else ""
            }
            Text(title, color = Color.White, style = MaterialTheme.typography.titleMedium)
            Text("Tap a ✓ angle to retake it", color = Color.White.copy(alpha = 0.7f), style = MaterialTheme.typography.bodySmall)
            Row {
                View.entries.forEach { view ->
                    val label = view.spoken().substringBefore(" ").replaceFirstChar { it.uppercase() }
                    FilterChip(
                        selected = view == session.wanted,
                        onClick = { onChoose(view) },
                        label = { Text(if (view in session.shots) "✓ $label" else label) },
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
            Row {
                if (session.mode == CaptureMode.GARMENT) {
                    OutlinedButton(onClick = onFinish) { Text("Finish", color = Color.White) }
                    Spacer(Modifier.width(8.dp))
                    Button(onClick = onNextGarment, enabled = session.complete) { Text("Next garment") }
                } else {
                    OutlinedButton(onClick = onFinish) { Text("Cancel setup", color = Color.White) }
                }
            }
        }
    }
}

@Composable
private fun TrayBadge(count: Int, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier, shape = RoundedCornerShape(16.dp),
        color = if (count > 0) MaterialTheme.colorScheme.primary else Color.Black.copy(alpha = 0.6f),
    ) {
        Text(
            "Review tray: $count", Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            color = if (count > 0) MaterialTheme.colorScheme.onPrimary else Color.White,
            style = MaterialTheme.typography.labelLarge,
        )
    }
}

@Composable
private fun HeightDialog(onSave: (Float) -> Unit, onCancel: () -> Unit) {
    var text by remember { mutableStateOf("") }
    val height = parseHeightCm(text)
    AlertDialog(
        onDismissRequest = {},
        title = { Text("Your height") },
        text = {
            Column {
                Text("Used to measure garment lengths. Centimetres, e.g. 175.")
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    label = { Text("Height (cm)") },
                    singleLine = true,
                    isError = text.isNotBlank() && height == null,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                )
            }
        },
        confirmButton = { TextButton(onClick = { height?.let(onSave) }, enabled = height != null) { Text("Save body figure") } },
        dismissButton = { TextButton(onClick = onCancel) { Text("Discard") } },
    )
}

@Composable
private fun PoseReadout(pose: PoseFrame?, check: PoseCheckResult?, modifier: Modifier = Modifier) {
    val text = when {
        pose == null -> "Pose: starting…"
        pose.landmarks.isEmpty() -> "Pose: nobody in view · ${pose.inferenceMs} ms"
        else -> "Pose: ${pose.inferenceMs} ms · seen ${check?.seenView?.spoken() ?: "?"}"
    }
    Surface(modifier = modifier, shape = RoundedCornerShape(8.dp), color = Color.Black.copy(alpha = 0.6f)) {
        Text(
            text, Modifier.padding(horizontal = 10.dp, vertical = 6.dp), color = Color.White,
            style = MaterialTheme.typography.bodySmall,
        )
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
private fun LastShotCard(shot: CameraViewModel.LastShot, modifier: Modifier = Modifier) {
    Surface(modifier = modifier, shape = RoundedCornerShape(12.dp), color = Color.Black.copy(alpha = 0.6f)) {
        Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Image(
                bitmap = shot.thumbnail,
                contentDescription = "Last photo",
                modifier = Modifier.height(96.dp).clip(RoundedCornerShape(8.dp)),
            )
            Spacer(Modifier.width(12.dp))
            Column {
                Text(
                    shot.view.spoken().replaceFirstChar { it.uppercase() }, color = Color.White,
                    style = MaterialTheme.typography.titleSmall,
                )
                Text(
                    "${shot.width} × ${shot.height}", color = Color.White.copy(alpha = 0.8f),
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
