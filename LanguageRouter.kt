package com.antonio.my.ai.girlfriend.free.amelia.bridge

import com.chaquo.python.Python

/**
 * LanguageRouter
 *
 * Smali-safe static entry point.
 * This is called directly from ChatActivity$e.smali.
 */
object LanguageRouter {

    @JvmStatic
    fun process(input: String): String {
        return try {
            val py = Python.getInstance()
            val module = py.getModule("language_pipeline")
            module.callAttr("process", input).toString()
        } catch (e: Exception) {
            // FAIL OPEN: never crash UI
            input
        }
    }
}
