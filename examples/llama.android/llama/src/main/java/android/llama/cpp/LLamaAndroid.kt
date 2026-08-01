package android.llama.cpp

import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    private val tag: String? = this::class.simpleName

    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("llama-android")

            // Set llama log handler to Android
            log_to_android()
            backend_init()

            Log.d(tag, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private val nlen: Int = 10

    private external fun log_to_android()
    private external fun load_model(filename: String): Long
    private external fun free_model(model: Long)
    private external fun new_context(model: Long): Long
    private external fun free_context(context: Long)
    private external fun backend_init()
    private external fun backend_free()
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun free_batch(batch: Long)
    private external fun new_sampler(): Long
    private external fun free_sampler(sampler: Long)
    private external fun bench_model(
        context: Long,
        model: Long,
        batch: Long,
        pp: Int,
        tg: Int,
        pl: Int,
        nr: Int
    ): String

    private external fun system_info(): String

    private external fun completion_init(
        context: Long,
        batch: Long,
        text: String,
        formatChat: Boolean,
        nLen: Int
    ): Int

    private external fun completion_loop(
        context: Long,
        batch: Long,
        sampler: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    // Multi-turn chat: prefill the user turn at the running KV position (after
    // the loaded .kv context + prior turns), routing through mzcache swapin when
    // the KV was swapped out. Does not clear the KV between turns.
    private external fun chat_completion_init(
        context: Long,
        model: Long,
        mzcache: Long,
        batch: Long,
        text: String
    ): Int

    private external fun chat_completion_loop(
        context: Long,
        model: Long,
        mzcache: Long,
        batch: Long,
        sampler: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    private external fun kv_cache_clear(context: Long)

    // Detokenized text of the loaded .kv context (empty if none loaded).
    private external fun loaded_context_text(): String

    //mzcache
    private external fun mz_init(filesDir: String)

    private external fun new_mzcache(context: Long, model: Long): Long

    private external fun free_mzcache(mzcache: Long)

    private external fun handleMemoryPressureNative(context: Long, model: Long, mzcache: Long)

    private external fun mz_decode(context: Long, model: Long, mzcache: Long)

    private external fun baseline_decode(context: Long, model: Long)

    // Readable from any thread (the authoritative State lives in a
    // ThreadLocal only visible on the run loop).
    @Volatile
    var isLoaded: Boolean = false
        private set

    suspend fun handleMemoryPressure() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    handleMemoryPressureNative(state.context, state.model, state.mzcache)
                }
                else -> {
                    Log.i(tag, "handleMemoryPressure called but no model loaded")
                }
            }
        }
    }

    suspend fun mz_run() {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    mz_decode(state.context, state.model, state.mzcache)
                }
                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun baseline_run() {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    baseline_decode(state.context, state.model)
                }
                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    Log.d(tag, "bench(): $state")
                    bench_model(state.context, state.model, state.batch, pp, tg, pl, nr)
                }

                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    /**
     * Loads the model unless one is already loaded. Returns true when this
     * call performed the load, false when a model was already loaded.
     * Runs on the single-threaded run loop, so check-then-load is race-free.
     */
    suspend fun ensureLoaded(pathToModel: String, filesDir: String): Boolean {
        return withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    // mzcache reads/writes its swap files relative to the CWD;
                    // move it to an app-writable directory before anything else.
                    mz_init(filesDir)

                    val model = load_model(pathToModel)
                    if (model == 0L)  throw IllegalStateException("load_model() failed")

                    val context = new_context(model)
                    if (context == 0L) throw IllegalStateException("new_context() failed")

                    val mzcacheCore = new_mzcache(context, model)
                    if (mzcacheCore == 0L) throw IllegalStateException("new_mzcache() failed")

                    val batch = new_batch(512, 0, 1)
                    if (batch == 0L) throw IllegalStateException("new_batch() failed")

                    val sampler = new_sampler()
                    if (sampler == 0L) throw IllegalStateException("new_sampler() failed")

                    Log.i(tag, "Loaded model $pathToModel")
                    threadLocalState.set(State.Loaded(model, context, mzcacheCore, batch, sampler))
                    isLoaded = true
                    true
                }
                else -> false
            }
        }
    }

    suspend fun load(pathToModel: String, filesDir: String) {
        if (!ensureLoaded(pathToModel, filesDir)) {
            throw IllegalStateException("Model already loaded")
        }
    }

    /** Detokenized text of the loaded .kv context, or "" if none is loaded. */
    suspend fun loadedContext(): String {
        return withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Loaded -> loaded_context_text()
                else -> ""
            }
        }
    }

    fun send(message: String, formatChat: Boolean = false): Flow<String> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                val ncur = IntVar(completion_init(state.context, state.batch, message, formatChat, nlen))
                while (ncur.value <= nlen) {
                    val str = completion_loop(state.context, state.batch, state.sampler, nlen, ncur)
                    if (str == null) {
                        break
                    }
                    emit(str)
                }
                kv_cache_clear(state.context)
            }
            else -> {}
        }
    }.flowOn(runLoop)

    /**
     * Multi-turn chat streaming. Unlike [send], the user turn is prefilled after
     * the loaded .kv context and every prior turn (the conversation accumulates),
     * the mzcache swapin path runs when the KV was swapped out, and the KV is not
     * cleared afterwards. Emits assistant token pieces as they are generated.
     */
    fun chat(message: String): Flow<String> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                chat_completion_init(state.context, state.model, state.mzcache, state.batch, message)
                val ncur = IntVar(0) // tokens generated this turn (capped at nlen)
                while (true) {
                    val str = chat_completion_loop(
                        state.context, state.model, state.mzcache, state.batch, state.sampler, nlen, ncur
                    ) ?: break
                    emit(str)
                }
            }
            else -> {}
        }
    }.flowOn(runLoop)

    /**
     * Unloads the model and frees resources.
     *
     * This is a no-op if there's no model loaded.
     */
    suspend fun unload() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    free_mzcache(state.mzcache)
                    free_context(state.context)
                    free_model(state.model)
                    free_batch(state.batch)
                    free_sampler(state.sampler);

                    threadLocalState.set(State.Idle)
                    isLoaded = false
                }
                else -> {}
            }
        }
    }

    companion object {
        private class IntVar(value: Int) {
            @Volatile
            var value: Int = value
                private set

            fun inc() {
                synchronized(this) {
                    value += 1
                }
            }
        }

        private sealed interface State {
            data object Idle: State
            data class Loaded(val model: Long, val context: Long, val mzcache: Long, val batch: Long, val sampler: Long): State
        }

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()

        fun instance(): LLamaAndroid = _instance
    }
}
