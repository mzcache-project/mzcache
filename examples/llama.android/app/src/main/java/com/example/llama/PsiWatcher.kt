package com.example.llama

import android.util.Log

class PsiWatcher {
    companion object {
        init {
            try {
                System.loadLibrary("psiwatcher")
            } catch (t: Throwable) {
                Log.e("PsiWatcher", "loadLibrary failed", t)
            }
        }
        @JvmStatic external fun nativeStart(cb: Callback, mode: Int, windowUs: Int, thresholdUs: Int): Boolean
        @JvmStatic external fun nativeStop()
    }

    /** mode: 0 = some, 1 = full */
    fun startSome(windowUs: Int = 200_000, thresholdUs: Int = 500_000, cb: Callback): Boolean =
        nativeStart(cb, /*mode=*/0, windowUs, thresholdUs)

    fun startFull(windowUs: Int = 1_000_000, thresholdUs: Int = 1_000_000, cb: Callback): Boolean =
        nativeStart(cb, /*mode=*/1, windowUs, thresholdUs)

    fun stop() = nativeStop()

    fun interface Callback {
        /** Called when a PSI event fires (not on the UI thread) */
        fun onPsiEvent(mode: Int, windowUs: Int, thresholdUs: Int)
    }
}