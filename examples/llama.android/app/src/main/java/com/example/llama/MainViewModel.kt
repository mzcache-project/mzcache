package com.example.llama

import android.llama.cpp.LLamaAndroid
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

enum class Role { User, Assistant }
data class ChatMessage(val role: Role, val text: String)

class MainViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    // Internal status log (experiment/auto flows). Not shown in the chat UI.
    var messages by mutableStateOf(listOf<String>())
        private set

    // Default prompt used when Send is pressed with an EMPTY Message field — so
    // the field stays clean/empty after loading (no confusing pre-fill), and
    // returning to the app + tapping Send fires this message straight away.
    // "Summarize context" (no article) is one token shorter than "Summarize the
    // context" -> 16 input tokens with the chat template + /no_think.
    val defaultPrompt = "Summarize context"

    var message by mutableStateOf("")
        private set

    // Chat bubbles shown in the UI (multi-turn conversation).
    var chat by mutableStateOf(listOf<ChatMessage>())
        private set

    // True while the model + KV state load into memory; the UI shows a
    // "Model and Context Loading…" indicator until it clears.
    var isLoading by mutableStateOf(false)
        private set

    var modelLoaded by mutableStateOf(false)
        private set

    // True once the loaded .kv context has been seeded into the chat history.
    private var contextSeeded = false

    // Path to the model file (Qwen3-0.6B), set by MainActivity.
    var modelPath: String = ""

    // App-internal files directory (set by MainActivity). mzcache chdir()s
    // here; must be internal storage because its file I/O uses O_DIRECT,
    // which the FUSE-backed external storage does not support.
    var filesDir: String = ""

    override fun onCleared() {
        super.onCleared()

        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                messages += exc.message!!
            }
        }
    }

    // Load model + KV state into memory (idempotent). Shows the loading
    // indicator; shared by the Load button and the first Send.
    private suspend fun ensureLoadedUi() {
        if (modelLoaded) return
        isLoading = true
        try {
            llamaAndroid.ensureLoaded(modelPath, filesDir)
            modelLoaded = true
            // Seed the chat with the restored .kv context (the loaded tokens
            // detokenized) so the conversation opens on the prefilled context.
            // Split into ~8000-char bubbles: a single Compose Text node cannot
            // exceed ~262144px, and the full 32700-token context is far taller,
            // so one bubble crashes layout. LazyColumn measures chunks lazily.
            // Prepend (not replace) so it stays first even if a Send raced the
            // load and already appended a user turn.
            val context = llamaAndroid.loadedContext()
            if (context.isNotBlank() && !contextSeeded) {
                val chunks = context.chunked(8000).map { ChatMessage(Role.Assistant, it) }
                chat = chunks + chat
                contextSeeded = true
            }
        } finally {
            isLoading = false
        }
    }

    // The field now starts empty, so there is no pre-filled prompt to clear on
    // focus; kept as a no-op hook for MainActivity's onFocusChanged.
    fun onInputFocused() {
    }

    fun onLoad() {
        viewModelScope.launch {
            try {
                ensureLoadedUi()
            } catch (exc: Exception) {
                Log.e(tag, "load failed", exc)
            }
        }
    }

    fun onSend() {
        if (isLoading) return
        // Mark the exact moment Send is pressed for the memory trace ("Request
        // arrived"), on the main thread before the coroutine/JNI hop to the
        // native prefill — otherwise the marker lands a bit late.
        Log.i(tag, "Request arrived: Send pressed")
        // Empty field -> send the default prompt (returning to the app + Send
        // fires "Summarize context" immediately); otherwise send what was typed.
        val text = message.trim().ifEmpty { defaultPrompt }
        message = ""   // leave the field empty after sending
        chat = chat + ChatMessage(Role.User, text)

        viewModelScope.launch {
            try {
                ensureLoadedUi()  // load-if-needed (app restart / not loaded yet)
                chat = chat + ChatMessage(Role.Assistant, "")
                // Multi-turn: prefills after the loaded context + prior turns and
                // routes through mzcache swapin when the KV was swapped out.
                llamaAndroid.chat(text)
                    .catch { Log.e(tag, "chat() failed", it); appendToLast(it.message ?: "error") }
                    .collect { appendToLast(it) }
            } catch (exc: Exception) {
                Log.e(tag, "onSend failed", exc)
                appendToLast(exc.message ?: "error")
            }
        }
    }

    private fun appendToLast(token: String) {
        val last = chat.lastOrNull() ?: return
        chat = chat.dropLast(1) + last.copy(text = last.text + token)
    }

    fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
        viewModelScope.launch {
            try {
                val start = System.nanoTime()
                val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
                val end = System.nanoTime()

                messages += warmupResult

                val warmup = (end - start).toDouble() / NanosPerSecond
                messages += "Warm up time: $warmup seconds, please wait..."

                if (warmup > 5.0) {
                    messages += "Warm up took too long, aborting benchmark"
                    return@launch
                }

                messages += llamaAndroid.bench(512, 128, 1, 3)
            } catch (exc: IllegalStateException) {
                Log.e(tag, "bench() failed", exc)
                messages += exc.message!!
            }
        }
    }

    fun load(pathToModel: String) {
        viewModelScope.launch {
            try {
                llamaAndroid.load(pathToModel, filesDir)
                messages += "Loaded $pathToModel"
            } catch (exc: IllegalStateException) {
                Log.e(tag, "load() failed", exc)
                messages += exc.message!!
            }
        }
    }

    fun updateMessage(newMessage: String) {
        message = newMessage
    }

    fun clear() {
        messages = listOf()
    }

    fun log(message: String) {
        messages += message
    }

    fun onMemoryPressure() {
        viewModelScope.launch {
            llamaAndroid.handleMemoryPressure() // runs safely on the native run loop
        }
    }

    fun onMzRun() {
        viewModelScope.launch {
            llamaAndroid.mz_run()
        }
    }

    // =========================
    //  adb-driven auto flow (`am start ... --ez auto true`)
    // =========================
    private val autoRunning = AtomicBoolean(false)

    // Baseline builds have no swapout, so the follow-up prefill is a plain
    // decode; mzcache builds go through swapin_generate.
    private suspend fun autoPrefill() {
        if (BuildConfig.MZ_BASELINE_NO_TRIM) {
            llamaAndroid.baseline_run()
        } else {
            llamaAndroid.mz_run()
        }
    }

    /**
     * One deterministic sequence for both the killed (cold: load model+state,
     * then prefill) and alive (warm: prefill only) cases. Cold/warm is decided
     * by whether ensureLoaded actually performed the load — after an LMK kill
     * the auto intent can still arrive via onNewIntent (task retained in
     * recents), so the callback that carried it says nothing.
     *
     * Harness contract (tag AUTO):
     *   AUTO_DONE case=cold pid=<p> load_ms=<X> prefill_ms=<Y> total_ms=<Z> recv_ms=<W>
     *   AUTO_DONE case=warm pid=<p> prefill_ms=<Y> total_ms=<Z>
     */
    fun runAuto(modelPath: String, drawGate: CompletableDeferred<Unit>, tReceiptMs: Long) {
        if (!autoRunning.compareAndSet(false, true)) {
            Log.i("AUTO", "AUTO_SKIP reason=already_running")
            return
        }
        viewModelScope.launch {
            try {
                Log.i("AUTO", "AUTO_START pid=${Process.myPid()}")

                val tLoad = SystemClock.elapsedRealtime()
                val cold = llamaAndroid.ensureLoaded(modelPath, filesDir)
                val loadMs = if (cold) SystemClock.elapsedRealtime() - tLoad else -1L
                if (cold) messages += "Loaded $modelPath (auto)"

                drawGate.await() // same after-first-draw semantics as runMzAfterDraw

                val tPre = SystemClock.elapsedRealtime()
                autoPrefill()
                val prefillMs = SystemClock.elapsedRealtime() - tPre

                val now = SystemClock.elapsedRealtime()
                if (cold) {
                    val start = Process.getStartElapsedRealtime()
                    Log.i("AUTO", "AUTO_DONE case=cold pid=${Process.myPid()} load_ms=$loadMs prefill_ms=$prefillMs total_ms=${now - start} recv_ms=${tReceiptMs - start}")
                } else {
                    Log.i("AUTO", "AUTO_DONE case=warm pid=${Process.myPid()} prefill_ms=$prefillMs total_ms=${now - tReceiptMs}")
                }
            } catch (e: Exception) {
                Log.e("AUTO", "AUTO_FAIL err=${e.message}", e)
            } finally {
                autoRunning.set(false)
            }
        }
    }
}
