package com.example.llama

import android.app.ActivityManager
import android.app.DownloadManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ComponentCallbacks2
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.StrictMode
import android.os.SystemClock
import android.text.format.Formatter
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme
import com.topjohnwu.superuser.Shell
import kotlinx.coroutines.CompletableDeferred
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Locale

// Process-wide guard for libsu's one-shot Shell.setDefaultBuilder (see onCreate).
private var shellBuilderInitialized = false

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
    clipboardManager: ClipboardManager? = null,
) : ComponentActivity(), ComponentCallbacks2 {
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels()

    // Model file path used by the adb-driven auto flow (set in onCreate).
    private lateinit var autoModelPath: String

    private fun readGpumemMappedBytesBySu(pid: Int): Long {
        val path = "/sys/devices/virtual/kgsl/kgsl/proc/$pid/gpumem_mapped"
        val res = Shell.cmd("cat $path").exec()
        return if (res.isSuccess && res.out.isNotEmpty())
            res.out[0].trim().toLongOrNull() ?: -1L
        else -1L
    }

    /** Device-wide memory status */
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { info -> activityManager.getMemoryInfo(info) }
    }

    // =========================
    //  CSV sampler (HandlerThread)
    // =========================
    private var memThread: HandlerThread? = null
    private var memHandler: Handler? = null
    private var csvWriter: BufferedWriter? = null
    private var csvFilePath: String? = null
    private val tsFmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private var t0: Long = 0L

    private val memSampleRunnable = object : Runnable {
        override fun run() {
            try {
                sampleAndWriteRow()
            } catch (t: Throwable) {
                Log.e(tag, "mem sample error", t)
            } finally {
                // repeat every 100ms
                memHandler?.postDelayed(this, 100L)
            }
        }
    }

    private fun openCsv() {
        val dir = getExternalFilesDir(null)!!
        val name = "mem_${System.currentTimeMillis()}.csv"
        val file = File(dir, name)
        csvFilePath = file.absolutePath
        csvWriter = BufferedWriter(FileWriter(file, false)).apply {
            write(
                "timestamp,elapsed_ms," +
                    "totalPss_kb,dalvikPss_kb,nativePss_kb,otherPss_kb," +
                    "totalPrivateDirty_kb,totalSharedDirty_kb," +
                    "totalRss_kb," +
                    "java_heap_used_kb,java_heap_free_kb,java_heap_total_kb," +
                    "system_avail_mem_kb,system_total_mem_kb," +
                    "gpumem_mapped_bytes\n"
            )
            flush()
        }
        Log.i(tag, "CSV: $csvFilePath")
        viewModel.log("CSV: $csvFilePath")
    }

    private fun sampleAndWriteRow() {
        val pid = android.os.Process.myPid()
        val memInfos = activityManager.getProcessMemoryInfo(intArrayOf(pid))
        val mi = memInfos[0]

        // values in kB
        val totalPss = mi.totalPss
        val dalvikPss = mi.dalvikPss
        val nativePss = mi.nativePss
        val otherPss = totalPss - dalvikPss - nativePss
        val totalPrivateDirty = mi.totalPrivateDirty
        val totalSharedDirty = mi.totalSharedDirty

        // May be provided on Android 13 (API 33)+ (0 if absent)
        val totalRss = runCatching { mi.memoryStats["totalRss"]?.toInt() ?: 0 }.getOrElse { 0 }

        val rt = Runtime.getRuntime()
        val javaUsed = ((rt.totalMemory() - rt.freeMemory()) / 1024L).toInt()
        val javaFree = (rt.freeMemory() / 1024L).toInt()
        val javaTotal = (rt.totalMemory() / 1024L).toInt()

        val sys = availableMemory()
        val sysAvailKb = (sys.availMem / 1024L).toInt()
        val sysTotalKb = (sys.totalMem / 1024L).toInt()
        val gpumemMappedBytes = readGpumemMappedBytesBySu(pid)

        val ts = tsFmt.format(java.util.Date())
        val elapsed = SystemClock.elapsedRealtime() - t0

        csvWriter?.apply {
            write(
                "$ts,$elapsed," +
                    "$totalPss,$dalvikPss,$nativePss,$otherPss," +
                    "$totalPrivateDirty,$totalSharedDirty," +
                    "$totalRss," +
                    "$javaUsed,$javaFree,$javaTotal," +
                    "$sysAvailKb,$sysTotalKb," +
                    "$gpumemMappedBytes\n"
            )
            // At a 100ms period, flushing immediately is fine (for long runs, flushing once per second is recommended)
            flush()
        }
    }

    private fun startMemSampler() {
        if (memThread != null) return
        t0 = SystemClock.elapsedRealtime()
        openCsv()
        memThread = HandlerThread("MemSampler").apply { start() }
        memHandler = Handler(memThread!!.looper)
        memHandler!!.post(memSampleRunnable)
    }

    private fun stopMemSampler() {
        try {
            memHandler?.removeCallbacksAndMessages(null)
            memThread?.quitSafely()
        } catch (_: Throwable) {
        } finally {
            memThread = null
            memHandler = null
        }
        try { csvWriter?.flush() } catch (_: Throwable) {}
        try { csvWriter?.close() } catch (_: Throwable) {}
        csvWriter = null
    }

    // =========================
    private var psi: PsiWatcher? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Shell.enableVerboseLogging = false
        // libsu's setDefaultBuilder() throws IllegalStateException ("The main shell
        // was already created") once a shell exists. onCreate runs again whenever the
        // Activity is relaunched (configuration change) in a still-live process, so
        // calling it unconditionally turned any relaunch into a FATAL that killed the
        // process — and with it a multi-GB loaded model + KV state. Configure once.
        if (!shellBuilderInitialized) {
            Shell.setDefaultBuilder(
                Shell.Builder.create().setFlags(Shell.FLAG_REDIRECT_STDERR)
            )
            shellBuilderInitialized = true
        }

        // StrictMode: minimal version (avoids compatibility issues)
        StrictMode.setVmPolicy(
            StrictMode.VmPolicy.Builder()
                .detectLeakedClosableObjects()
                .build()
        )

        // App-internal storage path that mzcache will chdir() into (O_DIRECT file I/O required)
        viewModel.filesDir = filesDir.absolutePath

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)

        viewModel.log("Current memory: $free / $total")
        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        // Model gguf and KV states are pre-staged on the device before launch
        // (see the AE prerequisites): the model is read from /data/local/tmp/gguf/
        // and the .kv states from /data/local/tmp/mzcache/states/ (the latter via
        // STATES_DIR in llama-android.cpp). The HuggingFace URL is kept only as a
        // reference for obtaining the gguf.
        val modelDir = File("/data/local/tmp/gguf")

        val models = listOf(
            Downloadable(
                "Qwen3-0.6B",
                Uri.parse("https://huggingface.co/appleyu/Qwen3-0.6B-FP16-gguf/resolve/main/Qwen3-0.6B-FP16.gguf?download=true"),
                File(modelDir, "Qwen3-0.6B-FP16.gguf"),
            ),
        )
        autoModelPath = models[0].destination.absolutePath

        // Start memory CSV sampling as soon as the app launches (100ms)
        startMemSampler()

        setContent {
            LlamaAndroidTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainCompose(
                        viewModel,
                        clipboardManager,
                        downloadManager,
                        models,
                        this
                    )
                }
            }
        }

        // psi = PsiWatcher()
        // val ok = psi!!.startSome(
        //     windowUs = 1_000_000,    // observation window 200ms
        //     thresholdUs = 500_000, // trigger if cumulative stall in that window exceeds 500ms
        //     cb = PsiWatcher.Callback { mode, win, thr ->
        //         Log.i(tag, "PSI event: mode=$mode winUs=$win thrUs=$thr")
        //         // event-driven action (no polling)
        //         viewModel.onMemoryPressure()
        //     }
        // )
        // Log.i(tag, "PSI watcher started=$ok")

        maybeStartAuto(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent) // Activity does not do this automatically
        maybeStartAuto(intent)
    }

    // Experiment harness entry point: `adb shell am start -n .../.MainActivity
    // --activity-single-top --ez auto true` triggers load-if-needed + 8-token
    // prefill with parseable AUTO_* logs, whether the process was killed or
    // not. After an LMK kill the task survives in recents, so the auto intent
    // may arrive via onNewIntent even on a cold start — runAuto classifies
    // cold/warm by whether the load actually ran, not by the callback.
    private fun maybeStartAuto(intent: Intent?) {
        if (intent?.getBooleanExtra("auto", false) != true) return
        intent.removeExtra("auto") // config-change recreation must not retrigger

        val tReceipt = SystemClock.elapsedRealtime()
        val drawGate = CompletableDeferred<Unit>()
        window.decorView.post { drawGate.complete(Unit) } // same gate as runMzAfterDraw

        viewModel.runAuto(autoModelPath, drawGate, tReceipt)
    }

    override fun onDestroy() {
        psi?.stop()
        psi = null
        stopMemSampler()
        super.onDestroy()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)

        if (BuildConfig.MZ_BASELINE_NO_TRIM) {
            Log.i(tag, "baseline: onTrimMemory(level=$level) ignored")
            return
        }

        // Ignore UI_HIDDEN — it fires on every backgrounding regardless of real
        // memory pressure. Evict only on the actual pressure levels
        // (RUNNING_MODERATE / LOW / CRITICAL, or BACKGROUND / MODERATE / COMPLETE).
        if (level == android.content.ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) {
            Log.i(tag, "onTrimMemory(UI_HIDDEN) ignored — no evict on backgrounding")
            return
        }
        Log.i(tag, "onTrimMemory called with level: $level")

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)
        Log.i(tag, "onTrimMemory starts: memory: $free / $total")

        viewModel.onMemoryPressure()

        val free2 = Formatter.formatFileSize(this, availableMemory().availMem)
        Log.i(tag, "onTrimMemory ends: memory: $free2 / $total")
    }

    override fun onLowMemory() {
        super.onLowMemory()

        if (BuildConfig.MZ_BASELINE_NO_TRIM) {
            Log.i(tag, "baseline: onLowMemory ignored")
            return
        }

        Log.i(tag, "onLowMemory called")
        viewModel.onMemoryPressure()
    }

    fun runMzAfterDraw(viewModel: MainViewModel) {
        window.decorView.post {
            val elapsed = SystemClock.elapsedRealtime() - t0
            Log.i(tag, "after button → elapsed=${elapsed}ms")
            viewModel.onMzRun()

            val afterDecode = SystemClock.elapsedRealtime() - t0
            Log.i(tag, "after button → after_decode=${afterDecode}ms")
        }
    }
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    dm: DownloadManager,
    models: List<Downloadable>,
    activity: MainActivity
) {
    // Model path for Load/Send (first entry = Qwen3-0.6B).
    LaunchedEffect(models) {
        models.firstOrNull()?.let { viewModel.modelPath = it.destination.path }
    }

    val focusManager = LocalFocusManager.current

    val scrollState = rememberLazyListState()
    // Keep the latest bubble in view as tokens stream in.
    LaunchedEffect(viewModel.chat.size, viewModel.chat.lastOrNull()?.text) {
        if (viewModel.chat.isNotEmpty()) scrollState.animateScrollToItem(viewModel.chat.size - 1)
    }

    Column(modifier = Modifier.fillMaxSize().padding(8.dp)) {
        // --- Chat history (~80%) ---
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LazyColumn(state = scrollState, modifier = Modifier.fillMaxSize()) {
                items(viewModel.chat) { msg -> ChatBubble(msg) }
            }
            if (viewModel.isLoading) {
                Row(
                    modifier = Modifier.align(Alignment.BottomCenter).padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text("Model and Context Loading…", style = MaterialTheme.typography.bodyMedium)
                }
            }
        }

        // --- Message input + Send icon ---
        Row(
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = viewModel.message,
                onValueChange = { viewModel.updateMessage(it) },
                label = { Text("Message") },
                modifier = Modifier
                    .weight(1f)
                    .onFocusChanged { if (it.isFocused) viewModel.onInputFocused() },
                maxLines = 4,
            )
            IconButton(
                onClick = { focusManager.clearFocus(); viewModel.onSend() },
                enabled = !viewModel.isLoading,
            ) {
                Icon(Icons.Filled.Send, contentDescription = "Send")
            }
        }

        // --- Load Model, KV cache ---
        Button(
            onClick = { viewModel.onLoad() },
            enabled = !viewModel.isLoading && !viewModel.modelLoaded,
            modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
        ) {
            Text(if (viewModel.modelLoaded) "Model Loaded" else "Load Model, KV cache")
        }
    }
}

@Composable
private fun ChatBubble(msg: ChatMessage) {
    val isUser = msg.role == Role.User
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp, horizontal = 4.dp),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            color = if (isUser) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.surfaceVariant,
            contentColor = if (isUser) MaterialTheme.colorScheme.onPrimary
                    else MaterialTheme.colorScheme.onSurfaceVariant,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.widthIn(max = 300.dp),
        ) {
            Text(
                text = msg.text.ifEmpty { "…" },
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                style = MaterialTheme.typography.bodyLarge,
            )
        }
    }
}
