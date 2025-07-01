package com.example.planarclassification

import android.annotation.SuppressLint
import android.app.Activity
import android.app.ActivityManager
import android.content.Context
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.widget.Toast
import com.example.planarclassification.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLContext
import javax.microedition.khronos.egl.EGLDisplay
import kotlin.concurrent.thread
import kotlin.math.*

class MainActivity : Activity() {
    companion object {
        private const val TAG = "NeuralPlayground"

        // Original dataset coordinate range
        private const val X_MIN = -4.5f
        private const val X_MAX = 4.0f
        private const val Y_MIN = -4.0f
        private const val Y_MAX = 4.0f

        // Benchmark test coordinates (circular pattern)
        private val benchmarkCoordinates = arrayOf(
            Pair(-3.0f, -2.0f), Pair(-2.0f, -3.0f), Pair(0.0f, -3.5f), Pair(2.0f, -3.0f),
            Pair(3.0f, -2.0f), Pair(3.5f, 0.0f), Pair(3.0f, 2.0f), Pair(2.0f, 3.0f),
            Pair(0.0f, 3.5f), Pair(-2.0f, 3.0f), Pair(-3.0f, 2.0f), Pair(-3.5f, 0.0f),
            Pair(-1.0f, -1.0f), Pair(1.0f, -1.0f), Pair(1.0f, 1.0f), Pair(-1.0f, 1.0f),
            Pair(0.0f, 0.0f), Pair(-2.5f, 0.0f), Pair(2.5f, 0.0f), Pair(0.0f, -2.5f),
            Pair(0.0f, 2.5f), Pair(-1.5f, -1.5f), Pair(1.5f, -1.5f), Pair(1.5f, 1.5f),
            Pair(-1.5f, 1.5f), Pair(-4.0f, 0.0f), Pair(3.5f, 0.0f), Pair(0.0f, -3.8f)
        )

        // 🎯 Ground Truth function (Coursera Deep Learning Week 3 - Circular pattern)
        fun getGroundTruth(x: Float, y: Float): Boolean {
            // General Coursera pattern: distance-based classification from center
            val distance = kotlin.math.sqrt(x * x + y * y)

            // Complex pattern: distance + angle combination (more realistic pattern)
            val angle = kotlin.math.atan2(y, x)

            // Pattern 1: Inner circle (distance < 1.5) → Blue
            if (distance < 1.5) return true

            // Pattern 2: Outer ring (distance > 3.0) → Blue
            if (distance > 3.0) return true

            // Pattern 3: Angle-based classification in middle region
            // First and third quadrants with specific conditions → Blue
            val isFirstOrThirdQuadrant = (x > 0 && y > 0) || (x < 0 && y < 0)
            if (distance >= 1.5 && distance <= 3.0) {
                return isFirstOrThirdQuadrant && (kotlin.math.abs(angle) < kotlin.math.PI / 3)
            }

            // Default: Red
            return false
        }

        // 🎯 Alternative Ground Truth functions (different patterns)
        fun getGroundTruthSimpleCircle(x: Float, y: Float): Boolean {
            // Simple circle: distance from center < 2.0 → Blue
            val distance = kotlin.math.sqrt(x * x + y * y)
            return distance < 2.0
        }

        fun getGroundTruthSpiral(x: Float, y: Float): Boolean {
            // Spiral pattern
            val distance = kotlin.math.sqrt(x * x + y * y)
            val angle = kotlin.math.atan2(y, x)
            val spiralValue = distance - 0.5 * angle
            return spiralValue > 0
        }
    }

    private lateinit var binding: ActivityMainBinding
    private var startTime: Long = 0
    private var inferenceTime: Long = 0
    private var firstFrame: Boolean = true
    private var tfliteModel: MappedByteBuffer? = null
    private var inputBuffer: ByteBuffer? = null

    // 🚀 Performance improvement: Interpreter reuse
    private var cpuInterpreter: Interpreter? = null
    private var gpuInterpreter: Interpreter? = null
    private var npuInterpreter: Interpreter? = null

    private var gpuDelegate: Any? = null
    private var nnApiDelegate: Any? = null
    private var isRunning = false
    private var isGpuAvailable = false
    private var isNnApiAvailable = false

    // Current coordinates (controlled by remote)
    private var currentX: Float = 0.0f
    private var currentY: Float = 0.0f
    private val moveStep = 0.2f

    // Classification result storage
    private val classificationHistory = mutableListOf<ClassificationPoint>()

    // 🔧 Safe Toast handler
    private val mainHandler = Handler(Looper.getMainLooper())

    // 📜 ScrollView reference (for auto-scroll)
    private lateinit var resultsScrollView: android.widget.ScrollView

    // 🎮 Enhanced navigation state management
    private var lastBackKeyTime: Long = 0
    private val BACK_KEY_TIMEOUT = 2000L // 2 seconds timeout for consecutive back presses
    private var backKeyPressCount = 0

    // Focus management (Clear button removed)
    private enum class FocusState {
        COORDINATE_AREA,     // Default state - coordinate manipulation
        BUTTON_CLASSIFY,
        BUTTON_BENCHMARK,
        BUTTON_NAVIGATE,     // Navigation button (Clear button removed)
        BUTTON_GPU_TOGGLE,   // 🔧 Fixed: Added GPU toggle to focus order
        RESULT_AREA_1,       // Left text area
        RESULT_AREA_2,       // Right text area
        VISUALIZATION_AREA   // Right panel - Neural Network Visualization
    }

    private var currentFocusState = FocusState.COORDINATE_AREA

    // 🔧 Fixed: Added BUTTON_GPU_TOGGLE to the focus order
    private val focusStateOrder = arrayOf(
        FocusState.COORDINATE_AREA,
        FocusState.BUTTON_CLASSIFY,
        FocusState.BUTTON_BENCHMARK,
        FocusState.BUTTON_NAVIGATE,
        FocusState.BUTTON_GPU_TOGGLE,  // 🔧 Fixed: Now included in navigation order
        FocusState.RESULT_AREA_1,
        FocusState.RESULT_AREA_2,
        FocusState.VISUALIZATION_AREA
    )

    data class ClassificationPoint(
        val x: Float,
        val y: Float,
        val probability: Float,
        val isBlue: Boolean,
        val inferenceTime: Long,
        val groundTruth: Boolean,  // Actual correct answer
        val isCorrect: Boolean     // Whether prediction is correct
    )

    data class BenchmarkResult(
        val avgInferenceTime: Long,
        val totalTime: Long,
        val backend: String,
        val pointCount: Int,
        val blueCount: Int,
        val redCount: Int
    )

    private var cpuBenchmarkResult: BenchmarkResult? = null
    private var gpuBenchmarkResult: BenchmarkResult? = null
    private var nnApiBenchmarkResult: BenchmarkResult? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)

            logDeviceInformation()
            logTensorFlowLiteInformation()
            checkOpenGLSupport()
            checkAcceleratorAvailability()

            setupUI()
            setupVisualization()

            // Load TensorFlow Lite model
            tfliteModel = try {
                loadModelFile("original_planar_classifier.tflite")
            } catch (e: IOException) {
                Log.e(TAG, "Failed to load TFLite model", e)
                showError("Failed to load TensorFlow Lite model")
                return
            }

            // 🚀 Performance improvement: Pre-allocate input buffer
            inputBuffer = ByteBuffer.allocateDirect(4 * 2).apply {
                order(ByteOrder.nativeOrder())
            }

            // 🚀 Performance improvement: Pre-create interpreters
            initializeInterpreters()

            Log.d(TAG, "🎉 Planar Classifier initialized successfully!")

        } catch (e: Exception) {
            Log.e(TAG, "Error during onCreate", e)
            showError("Failed to initialize app: ${e.message}")
        }
    }

    // 🚀 Performance improvement: Pre-create interpreters
    private fun initializeInterpreters() {
        try {
            Log.d(TAG, "🔧 Initializing interpreters...")

            // CPU interpreter
            val cpuOptions = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true) // 🚀 Enable XNNPACK
            }
            cpuInterpreter = Interpreter(tfliteModel!!, cpuOptions)
            Log.d(TAG, "✅ CPU interpreter initialized")

            // GPU interpreter
            if (isGpuAvailable) {
                try {
                    gpuDelegate = createGpuDelegate()
                    if (gpuDelegate != null) {
                        val gpuOptions = Interpreter.Options()
                        if (addDelegate(gpuOptions, gpuDelegate!!)) {
                            gpuInterpreter = Interpreter(tfliteModel!!, gpuOptions)
                            Log.d(TAG, "✅ GPU interpreter initialized")
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to initialize GPU interpreter: ${e.message}")
                }
            }

            // NPU interpreter
            if (isNnApiAvailable) {
                try {
                    nnApiDelegate = createNnApiDelegate()
                    if (nnApiDelegate != null) {
                        val npuOptions = Interpreter.Options()
                        if (addDelegate(npuOptions, nnApiDelegate!!)) {
                            npuInterpreter = Interpreter(tfliteModel!!, npuOptions)
                            Log.d(TAG, "✅ NPU interpreter initialized")
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to initialize NPU interpreter: ${e.message}")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing interpreters", e)
        }
    }

    // 🔍 Device hardware information logging (simplified)
    private fun logDeviceInformation() {
        Log.i(TAG, "========== DEVICE INFO ==========")
        Log.i(TAG, "Device: ${Build.MANUFACTURER} ${Build.MODEL}")
        Log.i(TAG, "Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        Log.i(TAG, "CPU: ${Build.SUPPORTED_ABIS.joinToString(", ")}")

        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val configInfo = activityManager.deviceConfigurationInfo
        Log.i(TAG, "OpenGL ES: ${configInfo.glEsVersion}")

        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        Log.i(TAG, "Memory: ${memoryInfo.availMem / (1024 * 1024)}MB / ${memoryInfo.totalMem / (1024 * 1024)}MB")
        Log.i(TAG, "================================")
    }

    // 🔍 TensorFlow Lite information check (simplified)
    private fun logTensorFlowLiteInformation() {
        Log.i(TAG, "========== TF LITE INFO ==========")

        try {
            val versionClass = Class.forName("org.tensorflow.lite.TensorFlowLite")
            val versionField = versionClass.getDeclaredField("VERSION_NAME")
            versionField.isAccessible = true
            val version = versionField.get(null) as? String ?: "Unknown"
            Log.i(TAG, "TF Lite Version: $version")
        } catch (e: Exception) {
            Log.w(TAG, "Could not get TF Lite version")
        }

        val classes = mapOf(
            "Interpreter" to "org.tensorflow.lite.Interpreter",
            "GpuDelegate" to "org.tensorflow.lite.gpu.GpuDelegate",
            "NnApiDelegate" to "org.tensorflow.lite.nnapi.NnApiDelegate"
        )

        classes.forEach { (name, className) ->
            try {
                Class.forName(className)
                Log.i(TAG, "✓ $name available")
            } catch (e: Exception) {
                Log.w(TAG, "✗ $name missing")
            }
        }
        Log.i(TAG, "=================================")
    }

    // 🔍 OpenGL support check (simplified)
    private fun checkOpenGLSupport() {
        Log.i(TAG, "========== OPENGL CHECK ==========")

        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val configInfo = activityManager.deviceConfigurationInfo
        val glVersion = configInfo.glEsVersion.toFloat()

        Log.i(TAG, "OpenGL ES Version: $glVersion")
        Log.i(TAG, "ES 2.0+: ${glVersion >= 2.0}")
        Log.i(TAG, "ES 3.0+: ${glVersion >= 3.0}")
        Log.i(TAG, "=================================")
    }

    // 🔍 Accelerator check (enhanced safety)
    private fun checkAcceleratorAvailability() {
        Log.i(TAG, "========== ACCELERATOR CHECK ==========")

        // GPU check
        isGpuAvailable = try {
            val gpuClass = Class.forName("org.tensorflow.lite.gpu.GpuDelegate")
            val constructor = gpuClass.getConstructor()
            val testDelegate = constructor.newInstance()
            testDelegate.javaClass.getMethod("close").invoke(testDelegate)
            Log.i(TAG, "✅ GPU acceleration available")
            true
        } catch (e: Exception) {
            Log.w(TAG, "❌ GPU not available: ${e.message}")
            false
        }

        // NPU check
        isNnApiAvailable = try {
            val nnApiClass = Class.forName("org.tensorflow.lite.nnapi.NnApiDelegate")
            val constructor = nnApiClass.getConstructor()
            val testDelegate = constructor.newInstance()
            testDelegate.javaClass.getMethod("close").invoke(testDelegate)
            Log.i(TAG, "✅ NPU acceleration available")
            true
        } catch (e: Exception) {
            Log.w(TAG, "❌ NPU not available: ${e.message}")
            false
        }

        Log.i(TAG, "Final Status - GPU: $isGpuAvailable, NPU: $isNnApiAvailable")
        Log.i(TAG, "======================================")
    }

    // 🔧 Safe GPU delegate creation
    private fun createGpuDelegate(): Any? {
        return if (isGpuAvailable) {
            try {
                val gpuClass = Class.forName("org.tensorflow.lite.gpu.GpuDelegate")
                gpuClass.getConstructor().newInstance()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create GPU delegate", e)
                null
            }
        } else null
    }

    // 🔧 Safe NNAPI delegate creation
    private fun createNnApiDelegate(): Any? {
        return if (isNnApiAvailable) {
            try {
                val nnApiClass = Class.forName("org.tensorflow.lite.nnapi.NnApiDelegate")
                nnApiClass.getConstructor().newInstance()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create NNAPI delegate", e)
                null
            }
        } else null
    }

    // Delegate addition
    private fun addDelegate(options: Interpreter.Options, delegate: Any): Boolean {
        return try {
            val delegateInterface = Class.forName("org.tensorflow.lite.Delegate")
            val addMethod = options.javaClass.getMethod("addDelegate", delegateInterface)
            addMethod.invoke(options, delegate)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add delegate", e)
            false
        }
    }

    // Delegate cleanup
    private fun closeDelegate(delegate: Any?) {
        delegate?.let {
            try {
                it.javaClass.getMethod("close").invoke(it)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to close delegate", e)
            }
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun setupUI() {
        binding.classifyButton.setOnClickListener { onClassifyClick(it) }
        binding.benchmarkButton.setOnClickListener { onBenchmarkClick(it) }
        binding.gpuToggle.setOnClickListener { onGPUClick(it) }
        binding.navigateButton.setOnClickListener { onNavigateClick(it) }

        // 🔍 ScrollView reference setup
        resultsScrollView = findViewById(R.id.resultsScrollView)

        // Disable accelerator toggle if no acceleration available
        binding.gpuToggle.isEnabled = isGpuAvailable || isNnApiAvailable

        // 🔄 Enable TextView scroll functionality
        binding.textView1.movementMethod = android.text.method.ScrollingMovementMethod()
        binding.textView2.movementMethod = android.text.method.ScrollingMovementMethod()

        // 🎮 Setup button focus management
        setupButtonFocus()

        // Initial UI state setup
        binding.textView1.text = """
            |🎯 Classification Results
            |
            |Ready to run planar classification!
            |
            |📱 Enhanced Navigation Controls:
            |• Use D-pad to move the cursor (coordinate area)
            |• Press Center button to classify current point
            |• Use navigation keys to move between controls
            |• 🎮 Nav button: Switch to visualization area
            |• 🚪 Exit key: Leave Results/Visualization → Focus Classify
            |• Results will show predicted vs actual class
            |• ✅ = Correct prediction, ❌ = Wrong prediction
            |
            |🚀 Features:
            |• Real-time neural network inference
            |• Hardware acceleration (CPU/NPU)
            |• Accuracy tracking and statistics
            |• Interactive coordinate visualization
            |• Enhanced navigation and focus management
            |• Dual-area cursor control
            |
            |📜 This area is scrollable - swipe up/down to see more content
        """.trimMargin()

        binding.textView2.text = when {
            isGpuAvailable && isNnApiAvailable -> """
                |📊 System Status - Android TV Ready
                |
                |🚀 GPU + NPU available
                |Hardware acceleration ready
                |
                |Device: SKB BMA-AI100 (4K TV Optimized)
                |Backends: CPU, GPU, NPU
                |
                |Ready for high-performance
                |neural network inference!
                |
                |🎯 Expected performance:
                |• CPU: ~1ms per inference
                |• NPU: ~2ms per inference
                |• High accuracy on test patterns
                |
                |🎮 TV Remote Navigation:
                |• D-pad: Move cursor (coordinate/visualization)
                |• Center: Execute action
                |• Navigation keys: Move between controls
                |• 🎮 Nav button: Switch to visualization
                |• Up/Down: Navigate or scroll
                |• 🚪 Exit key: Leave area → Focus Classify
                |• Results area: Exit key to exit
            """.trimMargin()
            isGpuAvailable -> """
                |📊 System Status - Android TV
                |
                |🎉 GPU available
                |Hardware acceleration enabled
                |
                |Backends: CPU, GPU
                |
                |Ready for accelerated
                |neural network inference!
                |
                |🎯 Expected performance:
                |• CPU: ~1ms per inference
                |• GPU: Variable performance
                |
                |🎮 TV Remote controls optimized
            """.trimMargin()
            isNnApiAvailable -> """
                |📊 System Status
                |
                |🧠 NPU available
                |Neural acceleration enabled
                |
                |Device: SKB BMA-AI100
                |Backends: CPU, NPU
                |
                |Ready for NPU-accelerated
                |neural network inference!
                |
                |🎯 Expected performance:
                |• CPU: ~1ms per inference
                |• NPU: ~2ms per inference
                |• Optimized for neural workloads
                |
                |🎮 Enhanced Navigation:
                |• D-pad: Move cursor (coordinate/visualization)
                |• Nav keys: Move between controls
                |• 🎮 Nav button: Switch to visualization
                |• Up/Down: Navigate or scroll
                |• 🚪 Exit key: Leave area → Focus Classify
                |• Results area: Exit key to exit
            """.trimMargin()
            else -> """
                |📊 System Status
                |
                |⚠ CPU only mode
                |No hardware acceleration
                |
                |Backend: CPU only
                |
                |Still functional, but slower
                |than hardware acceleration.
                |
                |🎯 Expected performance:
                |• CPU: ~1-2ms per inference
                |• No acceleration available
                |
                |🎮 Enhanced Navigation:
                |• Multiple control areas
                |• Seamless area switching
                |• Improved focus management
                |• Results area: Exit key to exit
            """.trimMargin()
        }

        binding.cpuBar.progress = 0
        binding.gpuBar.progress = 0

        // Set text colors
        binding.textView1.setTextColor(Color.BLACK)
        binding.textView2.setTextColor(Color.BLACK)

        updateCoordinateDisplay()
        updateFocusIndicator()
    }

    // 🎮 Setup button focus management for Android TV
    private fun setupButtonFocus() {
        // Make buttons focusable for TV remote navigation
        binding.classifyButton.isFocusable = true
        binding.benchmarkButton.isFocusable = true
        binding.gpuToggle.isFocusable = true
        binding.navigateButton.isFocusable = true

        // Enhanced TV focus behavior
        binding.classifyButton.isFocusableInTouchMode = false
        binding.benchmarkButton.isFocusableInTouchMode = false
        binding.gpuToggle.isFocusableInTouchMode = false
        binding.navigateButton.isFocusableInTouchMode = false

        // Make text areas focusable for scrolling
        binding.textView1.isFocusable = true
        binding.textView2.isFocusable = true
        binding.textView1.isFocusableInTouchMode = false
        binding.textView2.isFocusableInTouchMode = false

        // Make right panel focusable for TV navigation
        binding.coordinateView.isFocusable = true
        binding.coordinateView.isFocusableInTouchMode = true

        // TV-specific: Request focus on startup
        binding.classifyButton.requestFocus()

        Log.d(TAG, "🎮 Android TV button focus management initialized")
    }

    // 🎮 Update focus indicator visual feedback
    private fun updateFocusIndicator() {
        Log.d(TAG, "🎯 updateFocusIndicator() called - Setting focus to: ${currentFocusState.name}")

        // Reset all button backgrounds
        resetButtonHighlights()

        // Highlight current focused element
        when (currentFocusState) {
            FocusState.COORDINATE_AREA -> {
                // Highlight coordinate area with light green
                highlightCoordinateArea()
                Log.d(TAG, "🎯 Focus: Coordinate Area")
            }
            FocusState.BUTTON_CLASSIFY -> {
                Log.d(TAG, "🎯 Setting focus to Classify Button - requesting focus...")
                binding.classifyButton.requestFocus()
                highlightButton(binding.classifyButton)
                Log.d(TAG, "🎯 Focus: Classify Button - requestFocus() and highlight completed")
            }
            FocusState.BUTTON_BENCHMARK -> {
                binding.benchmarkButton.requestFocus()
                highlightButton(binding.benchmarkButton)
                Log.d(TAG, "🎯 Focus: Benchmark Button")
            }
            FocusState.BUTTON_GPU_TOGGLE -> {
                binding.gpuToggle.requestFocus()
                highlightButton(binding.gpuToggle)
                Log.d(TAG, "🎯 Focus: GPU Toggle Button")
            }
            FocusState.BUTTON_NAVIGATE -> {
                binding.navigateButton.requestFocus()
                highlightButton(binding.navigateButton)
                Log.d(TAG, "🎯 Focus: Navigate Button")
            }
            FocusState.RESULT_AREA_1 -> {
                binding.textView1.requestFocus()
                highlightTextArea(binding.textView1)
                Log.d(TAG, "🎯 Focus: Result Area 1 (Left)")
            }
            FocusState.RESULT_AREA_2 -> {
                binding.textView2.requestFocus()
                highlightTextArea(binding.textView2)
                Log.d(TAG, "🎯 Focus: Result Area 2 (Right)")
            }
            FocusState.VISUALIZATION_AREA -> {
                binding.coordinateView.requestFocus()
                highlightVisualizationArea()
                Log.d(TAG, "🎯 Focus: Visualization Area (Right Panel)")
            }
        }

        updateCoordinateDisplay()
        Log.d(TAG, "🎯 updateFocusIndicator() completed for: ${currentFocusState.name}")
    }

    private fun resetButtonHighlights() {
        // Android TV optimized: Use safe color access with fallback
        try {
            binding.classifyButton.setBackgroundColor(Color.parseColor("#4CAF50"))
            binding.benchmarkButton.setBackgroundColor(Color.parseColor("#FF9800"))
            binding.gpuToggle.setBackgroundColor(Color.parseColor("#607D8B"))
            binding.navigateButton.setBackgroundColor(Color.parseColor("#9C27B0"))
        } catch (e: Exception) {
            // Fallback colors for TV compatibility
            binding.classifyButton.setBackgroundColor(Color.GREEN)
            binding.benchmarkButton.setBackgroundColor(Color.rgb(255, 152, 0))
            binding.gpuToggle.setBackgroundColor(Color.GRAY)
            binding.navigateButton.setBackgroundColor(Color.rgb(156, 39, 176))
        }

        binding.textView1.setBackgroundColor(Color.parseColor("#F8F9FA"))
        binding.textView2.setBackgroundColor(Color.parseColor("#F8F9FA"))
        // 🔧 Ensure coordinate view always resets to white
        binding.coordinateView.setBackgroundColor(Color.WHITE)
        binding.coordinateText.setBackgroundColor(Color.parseColor("#E3F2FD"))
    }

    private fun highlightButton(button: View) {
        // Android TV optimized: Dark blue-grey highlight for better visibility
        button.setBackgroundColor(Color.parseColor("#455A64"))
    }

    private fun highlightTextArea(textView: View) {
        textView.setBackgroundColor(Color.parseColor("#ECEFF1"))
    }

    private fun highlightCoordinateArea() {
        binding.coordinateText.setBackgroundColor(Color.parseColor("#E8F5E8"))
    }

    private fun highlightVisualizationArea() {
        // 🔧 Don't change background color to avoid visual artifacts in ImageView
        // Just keep the white background and rely on text display for feedback
        // binding.coordinateView.setBackgroundColor(Color.parseColor("#E3F2FD"))

        // Alternative: could add a subtle border effect if needed
        // For now, just rely on the text indicator showing "🎮 NEURAL VISUALIZATION"
    }

    private fun setupVisualization() {
        drawCoordinateSystem()
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanupResources()
    }

    // 📜 Auto-scroll feature (scroll to top when content updates)
    private fun scrollToTop() {
        try {
            mainHandler.post {
                resultsScrollView.smoothScrollTo(0, 0)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Auto scroll failed: ${e.message}")
        }
    }

    // 📜 Auto-scroll feature (scroll to bottom when content updates)
    private fun scrollToBottom() {
        try {
            mainHandler.post {
                resultsScrollView.post {
                    resultsScrollView.fullScroll(android.view.View.FOCUS_DOWN)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Auto scroll to bottom failed: ${e.message}")
        }
    }

    private fun cleanupResources() {
        try {
            cpuInterpreter?.close()
            gpuInterpreter?.close()
            npuInterpreter?.close()
            closeDelegate(gpuDelegate)
            closeDelegate(nnApiDelegate)

            cpuInterpreter = null
            gpuInterpreter = null
            npuInterpreter = null
            gpuDelegate = null
            nnApiDelegate = null
            inputBuffer = null
            tfliteModel = null

            Log.d(TAG, "Resources cleaned up successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up resources", e)
        }
    }

    // 🎮 Enhanced remote control key event handling for Android TV
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        // 🔍 Debug: Log all key events
        Log.d(TAG, "onKeyDown - keyCode=$keyCode, currentFocus=${currentFocusState.name}")

        // 🔍 Special logging for potential Exit keys
        if (keyCode in 380..390 || keyCode == KeyEvent.KEYCODE_ESCAPE) {
            Log.i(TAG, "🚪 POTENTIAL EXIT KEY DETECTED: keyCode=$keyCode")
        }

        when (keyCode) {
            // D-pad controls for coordinate movement (only in coordinate area)
            KeyEvent.KEYCODE_DPAD_UP -> {
                if (currentFocusState == FocusState.COORDINATE_AREA || currentFocusState == FocusState.VISUALIZATION_AREA) {
                    Log.d(TAG, "🔍 D-pad UP - Current state: ${currentFocusState.name} - Moving cursor")
                    currentY = (currentY + moveStep).coerceAtMost(Y_MAX)
                    updateCoordinateDisplay()
                    drawCoordinateSystem()
                    return true
                } else {
                    Log.d(TAG, "🔍 D-pad UP - Current state: ${currentFocusState.name} - Navigation mode")
                    return handleNavigationUp()
                }
            }
            KeyEvent.KEYCODE_DPAD_DOWN -> {
                if (currentFocusState == FocusState.COORDINATE_AREA || currentFocusState == FocusState.VISUALIZATION_AREA) {
                    Log.d(TAG, "🔍 D-pad DOWN - Current state: ${currentFocusState.name} - Moving cursor")
                    currentY = (currentY - moveStep).coerceAtLeast(Y_MIN)
                    updateCoordinateDisplay()
                    drawCoordinateSystem()
                    return true
                } else {
                    Log.d(TAG, "🔍 D-pad DOWN - Current state: ${currentFocusState.name} - Navigation mode")
                    return handleNavigationDown()
                }
            }
            KeyEvent.KEYCODE_DPAD_LEFT -> {
                if (currentFocusState == FocusState.COORDINATE_AREA || currentFocusState == FocusState.VISUALIZATION_AREA) {
                    Log.d(TAG, "🔍 D-pad LEFT - Current state: ${currentFocusState.name} - Moving cursor")
                    currentX = (currentX - moveStep).coerceAtLeast(X_MIN)
                    updateCoordinateDisplay()
                    drawCoordinateSystem()
                    return true
                } else {
                    Log.d(TAG, "🔍 D-pad LEFT - Current state: ${currentFocusState.name} - Navigation mode")
                    return handleNavigationLeft()
                }
            }
            KeyEvent.KEYCODE_DPAD_RIGHT -> {
                if (currentFocusState == FocusState.COORDINATE_AREA || currentFocusState == FocusState.VISUALIZATION_AREA) {
                    Log.d(TAG, "🔍 D-pad RIGHT - Current state: ${currentFocusState.name} - Moving cursor")
                    currentX = (currentX + moveStep).coerceAtMost(X_MAX)
                    updateCoordinateDisplay()
                    drawCoordinateSystem()
                    return true
                } else {
                    Log.d(TAG, "🔍 D-pad RIGHT - Current state: ${currentFocusState.name} - Navigation mode")
                    return handleNavigationRight()
                }
            }

            // Action keys
            KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_ENTER -> {
                return handleCenterAction()
            }
            KeyEvent.KEYCODE_MEDIA_REWIND, KeyEvent.KEYCODE_MENU -> {
                return handleMenuKey()
            }
            KeyEvent.KEYCODE_MEDIA_FAST_FORWARD -> {
                toggleAccelerator()
                return true
            }

            // 🎮 Enhanced Back key behavior: Clear history → Focus to Classify button
            KeyEvent.KEYCODE_BACK -> {
                return handleBackKey()
            }

            // 🆕 Exit key (keyCode=385) for exiting specific areas - PRIMARY HANDLER
            385 -> {
                Log.d(TAG, "🚪 Exit key (385) detected in onKeyDown - calling handleExitKey()")
                return handleExitKey()
            }

            // 🔍 Test: Handle common alternative exit/back key codes
            KeyEvent.KEYCODE_ESCAPE -> {
                Log.d(TAG, "🚪 ESCAPE key detected - calling handleExitKey()")
                return handleExitKey()
            }

            // 🔍 Test: Handle unknown key codes that might be Exit key
            in 380..390 -> {
                Log.d(TAG, "🚪 Potential Exit key detected (keyCode=$keyCode) - calling handleExitKey()")
                return handleExitKey()
            }

            // 🔍 Temporary test: Use specific keys to test Exit functionality
            KeyEvent.KEYCODE_0 -> {
                if (currentFocusState == FocusState.VISUALIZATION_AREA || currentFocusState == FocusState.RESULT_AREA_1 || currentFocusState == FocusState.RESULT_AREA_2) {
                    Log.d(TAG, "🔍 TEST: Using '0' key as Exit key for testing")
                    return handleExitKey()
                }
                return false
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    // 🎮 Enhanced dispatchKeyEvent for better Exit key detection
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        // 🔍 Debug logging for key detection (only for ACTION_DOWN to avoid spam)
        if (event.action == KeyEvent.ACTION_DOWN) {
            Log.d(TAG, "dispatchKeyEvent - keyCode=${event.keyCode}, scanCode=${event.scanCode}, action=${event.action}, currentFocus=${currentFocusState.name}")
        }

        // Handle Exit key (keyCode=385) or alternative scan codes - but only if not already handled in onKeyDown
        if ((event.keyCode == 385 || event.scanCode == 174) && event.action == KeyEvent.ACTION_DOWN) {
            Log.d(TAG, "🚪 Exit key detected in dispatchKeyEvent (keyCode=${event.keyCode}, scanCode=${event.scanCode})")

            // Don't handle here if keyCode=385 (let onKeyDown handle it)
            if (event.keyCode == 385) {
                Log.d(TAG, "🔄 Exit key keyCode=385 - letting onKeyDown handle it")
                return super.dispatchKeyEvent(event)
            }

            // Handle alternative scan codes here
            Log.d(TAG, "🚪 Exit key alternative scanCode detected - calling handleExitKey()")
            val handled = handleExitKey()
            if (handled) {
                Log.d(TAG, "✅ Exit key handled successfully in dispatchKeyEvent")
                return true
            }
        }

        return super.dispatchKeyEvent(event)
    }

    // 🆕 Handle Exit key (keyCode=385) - Exit specific areas and return to Classify button
    private fun handleExitKey(): Boolean {
        Log.d(TAG, "🚪 handleExitKey() called - Current focus: ${currentFocusState.name}")

        when (currentFocusState) {
            FocusState.VISUALIZATION_AREA -> {
                Log.d(TAG, "🚪 Processing Exit from VISUALIZATION_AREA")

                // Exit visualization area and return to Classify button
                val previousState = currentFocusState
                currentFocusState = FocusState.BUTTON_CLASSIFY

                Log.d(TAG, "🚪 Focus changed: $previousState → ${currentFocusState.name}")

                // Force UI update on main thread
                runOnUiThread {
                    updateFocusIndicator()
                    updateCoordinateDisplay()

                    // Visual feedback - brief highlight
                    binding.classifyButton.requestFocus()
                    binding.classifyButton.setBackgroundColor(Color.parseColor("#2E7D32"))

                    Handler(Looper.getMainLooper()).postDelayed({
                        binding.classifyButton.setBackgroundColor(Color.parseColor("#4CAF50"))
                    }, 1000)
                }

                Log.d(TAG, "🚪 Exit key - Successfully exited Visualization area to Classify button")
                return true
            }
            FocusState.RESULT_AREA_1, FocusState.RESULT_AREA_2 -> {
                Log.d(TAG, "🚪 Processing Exit from RESULT_AREA (${currentFocusState.name})")

                // Exit Results & Analysis area and return to Classify button
                val previousState = currentFocusState
                currentFocusState = FocusState.BUTTON_CLASSIFY

                Log.d(TAG, "🚪 Focus changed: $previousState → ${currentFocusState.name}")

                // Force UI update on main thread
                runOnUiThread {
                    updateFocusIndicator()
                    updateCoordinateDisplay()

                    // Visual feedback - brief highlight
                    binding.classifyButton.requestFocus()
                    binding.classifyButton.setBackgroundColor(Color.parseColor("#2E7D32"))

                    Handler(Looper.getMainLooper()).postDelayed({
                        binding.classifyButton.setBackgroundColor(Color.parseColor("#4CAF50"))
                    }, 1000)
                }

                Log.d(TAG, "🚪 Exit key - Successfully exited Results area to Classify button")
                return true
            }
            FocusState.COORDINATE_AREA -> {
                // 🔧 FIXED: Handle exit from coordinate area (fallback for visualization area)
                Log.d(TAG, "🔧 Exit from COORDINATE_AREA - Moving to Classify button as fallback")
                currentFocusState = FocusState.BUTTON_CLASSIFY

                runOnUiThread {
                    updateFocusIndicator()
                    updateCoordinateDisplay()

                    // Visual feedback
                    binding.classifyButton.requestFocus()
                    binding.classifyButton.setBackgroundColor(Color.parseColor("#2E7D32"))

                    Handler(Looper.getMainLooper()).postDelayed({
                        binding.classifyButton.setBackgroundColor(Color.parseColor("#4CAF50"))
                    }, 1000)
                }

                Log.d(TAG, "🔧 Fallback: Moved from COORDINATE_AREA to Classify button")
                return true
            }
            else -> {
                Log.d(TAG, "🚪 Exit key - No action defined for ${currentFocusState.name}")
                return true
            }
        }
    }

    // 🆕 New: Menu key for running benchmark (kept for compatibility)
    private fun handleMenuKey(): Boolean {
        // Menu key now only runs benchmark, Exit key handles area navigation
        if (!isRunning) {
            runBenchmark()
            Log.d(TAG, "📱 Menu key - Running benchmark from ${currentFocusState.name}")
        } else {
            Log.d(TAG, "📱 Menu key - Benchmark already running")
        }
        return true
    }

    // 🎮 Handle Back key with consecutive press logic (Visualization area now uses Exit key only for exit)
    private fun handleBackKey(): Boolean {
        // 🔧 Results & Analysis area now requires Exit key to exit (not Back key)
        if (currentFocusState == FocusState.RESULT_AREA_1 || currentFocusState == FocusState.RESULT_AREA_2) {
            // Don't show toast message to avoid UI clutter
            Log.d(TAG, "🔙 Back key - Results area requires Exit key to exit")
            return true
        }

        // 🔧 Visualization area: Back key only clears history, doesn't exit (use Exit key to exit)
        if (currentFocusState == FocusState.VISUALIZATION_AREA) {
            Log.d(TAG, "🔙 Back key - Visualization area: clearing history only (use Exit key to exit)")

            val currentTime = SystemClock.uptimeMillis()

            if (currentTime - lastBackKeyTime > BACK_KEY_TIMEOUT) {
                // First back press: Clear classification history
                backKeyPressCount = 1
                lastBackKeyTime = currentTime
                clearClassificationHistory()
                Log.d(TAG, "🔙 Back key - Visualization area: History cleared")
                return true
            } else {
                // Second press within timeout: Still just clear (don't exit)
                backKeyPressCount++
                lastBackKeyTime = currentTime
                clearClassificationHistory()
                Log.d(TAG, "🔙 Back key - Visualization area: History cleared again (use Exit key to exit)")
                return true
            }
        }

        // Normal Back key handling for other areas
        val currentTime = SystemClock.uptimeMillis()

        if (currentTime - lastBackKeyTime > BACK_KEY_TIMEOUT) {
            // First back press or timeout occurred - reset counter
            backKeyPressCount = 1
            lastBackKeyTime = currentTime

            // First press: Clear classification history
            clearClassificationHistory()
            Log.d(TAG, "🔙 Back key - First press: History cleared")
            return true

        } else {
            // Second press within timeout
            backKeyPressCount++
            lastBackKeyTime = currentTime

            if (backKeyPressCount == 2) {
                // Second press: Move focus to Classify button
                currentFocusState = FocusState.BUTTON_CLASSIFY
                updateFocusIndicator()
                Log.d(TAG, "🔙 Back key - Second press: Focused Classify button")
                return true
            }
        }

        return true
    }

    // 🎮 Handle center/enter action based on current focus - Android TV optimized
    private fun handleCenterAction(): Boolean {
        return when (currentFocusState) {
            FocusState.COORDINATE_AREA -> {
                if (!isRunning) classifyCurrentPoint()
                true
            }
            FocusState.BUTTON_CLASSIFY -> {
                onClassifyClick(binding.classifyButton)
                true
            }
            FocusState.BUTTON_BENCHMARK -> {
                onBenchmarkClick(binding.benchmarkButton)
                true
            }
            FocusState.BUTTON_GPU_TOGGLE -> {
                onGPUClick(binding.gpuToggle)
                true
            }
            FocusState.BUTTON_NAVIGATE -> {
                onNavigateClick(binding.navigateButton)
                true
            }
            FocusState.RESULT_AREA_1, FocusState.RESULT_AREA_2 -> {
                // Center action in text areas - silent operation
                true
            }
            FocusState.VISUALIZATION_AREA -> {
                // Center action in visualization area - classify using standard method
                if (!isRunning) {
                    classifyCurrentPoint()
                }
                true
            }
        }
    }

    // 🎮 Navigation handlers - Android TV optimized with proper direction flow
    private fun handleNavigationUp(): Boolean {
        return when (currentFocusState) {
            FocusState.RESULT_AREA_1 -> {
                // Scroll up in left text area
                binding.textView1.scrollBy(0, -50)
                true
            }
            FocusState.RESULT_AREA_2 -> {
                // Scroll up in right text area
                binding.textView2.scrollBy(0, -50)
                true
            }
            FocusState.VISUALIZATION_AREA -> {
                // In visualization area, move cursor up
                currentY = (currentY + moveStep).coerceAtMost(Y_MAX)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                true
            }
            FocusState.COORDINATE_AREA, FocusState.BUTTON_CLASSIFY, FocusState.BUTTON_BENCHMARK,
            FocusState.BUTTON_GPU_TOGGLE, FocusState.BUTTON_NAVIGATE -> {
                // Move to previous focus state
                val currentIndex = focusStateOrder.indexOf(currentFocusState)
                if (currentIndex > 0) {
                    currentFocusState = focusStateOrder[currentIndex - 1]
                    updateFocusIndicator()
                }
                true
            }
        }
    }

    private fun handleNavigationDown(): Boolean {
        return when (currentFocusState) {
            FocusState.RESULT_AREA_1 -> {
                // Scroll down in left text area
                binding.textView1.scrollBy(0, 50)
                true
            }
            FocusState.RESULT_AREA_2 -> {
                // Scroll down in right text area
                binding.textView2.scrollBy(0, 50)
                true
            }
            FocusState.VISUALIZATION_AREA -> {
                // In visualization area, move cursor down
                currentY = (currentY - moveStep).coerceAtLeast(Y_MIN)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                true
            }
            FocusState.COORDINATE_AREA, FocusState.BUTTON_CLASSIFY, FocusState.BUTTON_BENCHMARK,
            FocusState.BUTTON_GPU_TOGGLE, FocusState.BUTTON_NAVIGATE -> {
                // Move to next focus state
                val currentIndex = focusStateOrder.indexOf(currentFocusState)
                if (currentIndex < focusStateOrder.size - 1) {
                    currentFocusState = focusStateOrder[currentIndex + 1]
                    updateFocusIndicator()
                }
                true
            }
        }
    }

    private fun handleNavigationLeft(): Boolean {
        return when (currentFocusState) {
            FocusState.RESULT_AREA_2 -> {
                // Move from right text area to left text area
                currentFocusState = FocusState.RESULT_AREA_1
                updateFocusIndicator()
                true
            }
            FocusState.BUTTON_BENCHMARK -> {
                // Move from Benchmark to Classify button
                currentFocusState = FocusState.BUTTON_CLASSIFY
                updateFocusIndicator()
                true
            }
            FocusState.BUTTON_NAVIGATE -> {
                // Move from Navigate to Benchmark button
                currentFocusState = FocusState.BUTTON_BENCHMARK
                updateFocusIndicator()
                true
            }
            FocusState.VISUALIZATION_AREA -> {
                // In visualization area, move cursor left
                currentX = (currentX - moveStep).coerceAtLeast(X_MIN)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                true
            }
            FocusState.COORDINATE_AREA, FocusState.BUTTON_CLASSIFY, FocusState.BUTTON_GPU_TOGGLE,
            FocusState.RESULT_AREA_1 -> {
                // No horizontal navigation for these states
                false
            }
        }
    }

    private fun handleNavigationRight(): Boolean {
        return when (currentFocusState) {
            FocusState.RESULT_AREA_1 -> {
                // Move from left text area to right text area
                currentFocusState = FocusState.RESULT_AREA_2
                updateFocusIndicator()
                true
            }
            FocusState.BUTTON_CLASSIFY -> {
                // Move from Classify to Benchmark button
                currentFocusState = FocusState.BUTTON_BENCHMARK
                updateFocusIndicator()
                true
            }
            FocusState.BUTTON_BENCHMARK -> {
                // Move from Benchmark to Navigate button
                currentFocusState = FocusState.BUTTON_NAVIGATE
                updateFocusIndicator()
                true
            }
            FocusState.VISUALIZATION_AREA -> {
                // In visualization area, move cursor right
                currentX = (currentX + moveStep).coerceAtMost(X_MAX)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                true
            }
            FocusState.COORDINATE_AREA, FocusState.BUTTON_GPU_TOGGLE, FocusState.BUTTON_NAVIGATE,
            FocusState.RESULT_AREA_2 -> {
                // No horizontal navigation for these states
                false
            }
        }
    }

    private fun toggleAccelerator() {
        Log.d(TAG, "📱 Remote control accelerator toggle triggered")

        if (!isGpuAvailable && !isNnApiAvailable) {
            Log.w(TAG, "No hardware acceleration available for toggle")
            return
        }

        val newState = !binding.gpuToggle.isChecked
        binding.gpuToggle.isChecked = newState
        Log.d(TAG, "🔄 Accelerator toggled via remote to: $newState")

        onGPUClick(binding.gpuToggle)
    }

    fun onClassifyClick(v: View) {
        if (isRunning) {
            Log.d(TAG, "Already running classification")
            return
        }
        classifyCurrentPoint()
    }

    fun onBenchmarkClick(v: View) {
        if (isRunning) {
            Log.d(TAG, "Already running benchmark")
            return
        }
        runBenchmark()
    }

    fun onNavigateClick(v: View) {
        // Switch focus to visualization area
        Log.d(TAG, "🎮 Navigate button clicked - Current state before: ${currentFocusState.name}")
        currentFocusState = FocusState.VISUALIZATION_AREA
        Log.d(TAG, "🎮 Navigate button clicked - Current state after: ${currentFocusState.name}")
        updateFocusIndicator()
        Log.d(TAG, "🎮 Navigate button clicked - Switched to visualization area - updateFocusIndicator completed")
    }

    // 🚀 Performance improvement: Use pre-created interpreters
    private fun classifyCurrentPoint() {
        if (tfliteModel == null) {
            showError("TensorFlow Lite model not loaded")
            return
        }

        thread {
            isRunning = true
            try {
                val useAccelerator = binding.gpuToggle.isChecked && (isGpuAvailable || isNnApiAvailable)

                Log.d(TAG, "🎯 Starting classification - Accelerator requested: $useAccelerator")
                Log.d(TAG, "🔍 Available interpreters - CPU: ${cpuInterpreter != null}, GPU: ${gpuInterpreter != null}, NPU: ${npuInterpreter != null}")

                val (interpreter, actualBackend) = when {
                    useAccelerator && npuInterpreter != null -> {
                        Log.d(TAG, "🧠 Using NPU interpreter for classification")
                        Pair(npuInterpreter!!, "NPU")
                    }
                    useAccelerator && gpuInterpreter != null -> {
                        Log.d(TAG, "🎉 Using GPU interpreter for classification")
                        Pair(gpuInterpreter!!, "GPU")
                    }
                    cpuInterpreter != null -> {
                        Log.d(TAG, "💻 Using CPU interpreter for classification")
                        Pair(cpuInterpreter!!, "CPU")
                    }
                    else -> {
                        Log.e(TAG, "❌ No interpreter available!")
                        throw IllegalStateException("No interpreter available")
                    }
                }

                Log.d(TAG, "⚡ Executing classification on $actualBackend at point ($currentX, $currentY)")
                val result = classifyPointFast(interpreter, currentX, currentY)

                result?.let { point ->
                    classificationHistory.add(point)
                    Log.d(TAG, "📊 Classification completed - Result: ${if (point.isBlue) "Blue" else "Red"}, Confidence: ${point.probability}, Time: ${point.inferenceTime}ms")

                    runOnUiThread {
                        val className = if (point.isBlue) "Blue" else "Red"
                        val confidence = if (point.isBlue) point.probability else (1f - point.probability)
                        val groundTruthClass = if (point.groundTruth) "Blue" else "Red"
                        val correctnessIcon = if (point.isCorrect) "✅" else "❌"
                        val correctnessText = if (point.isCorrect) "Correct" else "Wrong"

                        binding.textView1.text = """
                                |🎯 Latest Classification Result:
                                |
                                |📍 Point: (${String.format("%.2f", point.x)}, ${String.format("%.2f", point.y)})
                                |
                                |🤖 Predicted: $className (${String.format("%.1f", confidence * 100)}%)
                                |🎯 Actual: $groundTruthClass
                                |📊 Result: $correctnessIcon $correctnessText
                                |
                                |⚡ Inference: ${point.inferenceTime}ms ($actualBackend)
                                |
                                |━━━━━━━━━━━━━━━━━━━━━━━━
                                |
                                |📊 Performance Details:
                                |• Coordinates: Neural network input
                                |• Prediction: Model output classification
                                |• Ground Truth: Actual correct answer
                                |• Accuracy: Prediction correctness
                                |• Backend: ${actualBackend} acceleration
                                |
                                |🧠 How it works:
                                |• Input: 2D coordinates (x, y)
                                |• Processing: Neural network inference
                                |• Output: Binary classification (Blue/Red)
                                |• Validation: Compare with ground truth
                                |
                                |🎮 Current Focus: ${currentFocusState.name}
                            """.trimMargin()

                        // Calculate overall accuracy
                        val totalPoints = classificationHistory.size
                        val correctPredictions = classificationHistory.count { it.isCorrect }
                        val accuracy = if (totalPoints > 0) {
                            (correctPredictions * 100.0 / totalPoints)
                        } else 0.0

                        // Class-wise statistics
                        val bluePoints = classificationHistory.filter { it.groundTruth }
                        val redPoints = classificationHistory.filter { !it.groundTruth }
                        val blueAccuracy = if (bluePoints.isNotEmpty()) {
                            bluePoints.count { it.isCorrect } * 100.0 / bluePoints.size
                        } else 0.0
                        val redAccuracy = if (redPoints.isNotEmpty()) {
                            redPoints.count { it.isCorrect } * 100.0 / redPoints.size
                        } else 0.0

                        binding.textView2.text = """
                                |📊 Classification History & Statistics
                                |
                                |📈 Overall Performance:
                                |• Total Points: $totalPoints
                                |• Correct Predictions: $correctPredictions
                                |• Overall Accuracy: ${String.format("%.1f", accuracy)}%
                                |
                                |🎨 Class Distribution:
                                |• Blue Predictions: ${classificationHistory.count { it.isBlue }}
                                |• Red Predictions: ${classificationHistory.count { !it.isBlue }}
                                |
                                |🎯 Class-wise Accuracy:
                                |• Blue Class: ${String.format("%.1f", blueAccuracy)}% (${bluePoints.count { it.isCorrect }}/${bluePoints.size})
                                |• Red Class: ${String.format("%.1f", redAccuracy)}% (${redPoints.count { it.isCorrect }}/${redPoints.size})
                                |
                                |⚡ Current Backend: $actualBackend
                                |
                                |━━━━━━━━━━━━━━━━━━━━━━━━
                                |
                                |📝 Legend:
                                |• ● Filled circle = Correct prediction
                                |• ○ Empty circle + X = Wrong prediction
                                |• Blue/Red = Predicted class color
                                |
                                |🎮 Enhanced Navigation Controls:
                                |• Focus: ${currentFocusState.name}
                                |• D-pad: Move cursor (coordinate/visualization areas)
                                |• Nav keys: Move between buttons
                                |• Up/Down in text areas: Scroll content
                                |• Center: Execute current action
                                |• 🎮 Nav button: Switch to visualization area
                                |• 🚪 Exit key: Leave area → Focus Classify
                                |• Results area: Exit key to exit
                            """.trimMargin()

                        drawCoordinateSystem()

                        // 📜 Scroll to top when new result is displayed
                        scrollToTop()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ Error during classification", e)
                runOnUiThread {
                    showError("Classification failed: ${e.message}")
                }
            } finally {
                isRunning = false
                Log.d(TAG, "✅ Classification operation completed")
            }
        }
    }

    // 🚀 Performance optimized classification function (with warm-up + accuracy check)
    private fun classifyPointFast(interpreter: Interpreter, x: Float, y: Float, isWarmup: Boolean = false): ClassificationPoint? {
        return try {
            // Input preparation (buffer reuse)
            inputBuffer?.rewind()
            inputBuffer?.putFloat(x)
            inputBuffer?.putFloat(y)

            // Output preparation
            val outputBuffer = ByteBuffer.allocateDirect(4).apply {
                order(ByteOrder.nativeOrder())
            }

            // High precision timing (no measurement during warm-up)
            val startTime = if (!isWarmup) System.nanoTime() else 0L
            interpreter.run(inputBuffer, outputBuffer)
            val endTime = if (!isWarmup) System.nanoTime() else 0L

            val inferenceTimeMs = if (!isWarmup) {
                maxOf(1L, (endTime - startTime) / 1_000_000L)
            } else {
                1L // No timing measurement for warm-up
            }

            // Result extraction
            outputBuffer.rewind()
            val probability = outputBuffer.float
            val predictedIsBlue = probability >= 0.5f

            // 🎯 Ground Truth calculation and accuracy check
            val actualIsBlue = getGroundTruth(x, y)
            val isCorrect = predictedIsBlue == actualIsBlue

            ClassificationPoint(x, y, probability, predictedIsBlue, inferenceTimeMs, actualIsBlue, isCorrect)

        } catch (e: Exception) {
            if (!isWarmup) {
                Log.e(TAG, "Error classifying point ($x, $y)", e)
            }
            null
        }
    }

    private fun runBenchmark() {
        thread {
            isRunning = true
            try {
                runOnUiThread {
                    binding.textView1.text = "🚀 Running Performance Benchmark..."
                    binding.textView2.text = "Testing CPU performance..."
                    binding.cpuBar.progress = 0
                    binding.gpuBar.progress = 0
                }

                // CPU benchmark
                Log.d(TAG, "🚀 Starting CPU benchmark")
                val cpuStartTime = SystemClock.uptimeMillis()
                val cpuResults = runBenchmarkForBackend("CPU")
                val cpuTotalTime = SystemClock.uptimeMillis() - cpuStartTime

                cpuBenchmarkResult = createBenchmarkResult(cpuResults, cpuTotalTime, "CPU")
                Log.d(TAG, "🏁 CPU benchmark completed: ${cpuResults.size} points, avg: ${cpuBenchmarkResult?.avgInferenceTime}ms")

                // GPU benchmark
                if (isGpuAvailable && gpuInterpreter != null) {
                    runOnUiThread {
                        binding.textView1.text = "Testing GPU acceleration..."
                        binding.textView2.text = "GPU performance benchmark in progress..."
                    }

                    Thread.sleep(500)
                    Log.d(TAG, "🚀 Starting GPU benchmark")
                    val gpuStartTime = SystemClock.uptimeMillis()
                    val gpuResults = runBenchmarkForBackend("GPU")
                    val gpuTotalTime = SystemClock.uptimeMillis() - gpuStartTime

                    gpuBenchmarkResult = createBenchmarkResult(gpuResults, gpuTotalTime, "GPU")
                    Log.d(TAG, "🏁 GPU benchmark completed: ${gpuResults.size} points, avg: ${gpuBenchmarkResult?.avgInferenceTime}ms")
                }

                // NPU benchmark
                if (isNnApiAvailable && npuInterpreter != null) {
                    runOnUiThread {
                        binding.textView1.text = "Testing NPU acceleration..."
                        binding.textView2.text = "Neural Processing Unit benchmark..."
                    }

                    Thread.sleep(500)
                    Log.d(TAG, "🚀 Starting NPU benchmark")
                    val nnApiStartTime = SystemClock.uptimeMillis()
                    val nnApiResults = runBenchmarkForBackend("NPU")
                    val nnApiTotalTime = SystemClock.uptimeMillis() - nnApiStartTime

                    nnApiBenchmarkResult = createBenchmarkResult(nnApiResults, nnApiTotalTime, "NPU")
                    Log.d(TAG, "🏁 NPU benchmark completed: ${nnApiResults.size} points, avg: ${nnApiBenchmarkResult?.avgInferenceTime}ms")
                }

                showBenchmarkComparison()

            } catch (e: Exception) {
                Log.e(TAG, "Error during benchmark", e)
                runOnUiThread {
                    showError("Benchmark failed: ${e.message}")
                }
            } finally {
                isRunning = false
            }
        }
    }

    private fun createBenchmarkResult(results: List<ClassificationPoint>, totalTime: Long, backend: String): BenchmarkResult {
        val avgInference = if (results.isNotEmpty()) {
            val actualAverage = results.map { it.inferenceTime }.average()
            val roundedAverage = kotlin.math.round(actualAverage).toLong()

            Log.d(TAG, "📊 $backend Final Result: Actual avg=${String.format("%.1f", actualAverage)}ms, Rounded avg=${roundedAverage}ms (all values included)")

            roundedAverage
        } else {
            Log.w(TAG, "📊 $backend Final Result: No results to calculate average")
            0L
        }

        return BenchmarkResult(
            avgInferenceTime = avgInference,
            totalTime = totalTime,
            backend = backend,
            pointCount = results.size,
            blueCount = results.count { it.isBlue },
            redCount = results.count { !it.isBlue }
        )
    }

    // 🚀 Performance improvement: Use pre-created interpreters
    private fun runBenchmarkForBackend(backend: String): List<ClassificationPoint> {
        val results = mutableListOf<ClassificationPoint>()

        val interpreter = when (backend) {
            "GPU" -> gpuInterpreter
            "NPU" -> npuInterpreter
            else -> cpuInterpreter
        }

        if (interpreter == null) {
            Log.w(TAG, "$backend interpreter not available")
            return results
        }

        Log.d(TAG, "🔧 Running $backend benchmark with pre-initialized interpreter")

        // 🔥 Warm-up run for ALL backends to eliminate cold start overhead
        Log.d(TAG, "🔥 Performing warm-up run for $backend...")
        try {
            val warmupPoint = classifyPointFast(interpreter, 0.0f, 0.0f, isWarmup = true)
            if (warmupPoint != null) {
                Log.d(TAG, "🔥 Warm-up completed: ready for benchmark (initialization overhead eliminated)")
            }
            // Additional warm-up for CPU to ensure JIT optimization
            if (backend == "CPU") {
                repeat(3) {
                    classifyPointFast(interpreter, 1.0f, 1.0f, isWarmup = true)
                }
                Log.d(TAG, "🔥 Additional CPU warm-up completed (JIT optimization)")
            }
            Thread.sleep(100) // Small delay to ensure warm-up is complete
        } catch (e: Exception) {
            Log.w(TAG, "Warm-up failed for $backend: ${e.message}")
        }

        benchmarkCoordinates.forEachIndexed { index, (x, y) ->
            try {
                val point = classifyPointFast(interpreter, x, y)
                if (point != null) {
                    results.add(point)

                    // Log individual point timings (including accuracy)
                    val className = if (point.isBlue) "Blue" else "Red"
                    val groundTruthClass = if (point.groundTruth) "Blue" else "Red"
                    val correctIcon = if (point.isCorrect) "✅" else "❌"
                    Log.v(TAG, "📊 $backend Point #${index + 1}: ($x, $y) → $className (${String.format("%.3f", point.probability)}) vs $groundTruthClass $correctIcon - ${point.inferenceTime}ms")
                }

                val progress = ((index + 1) * 100) / benchmarkCoordinates.size
                runOnUiThread {
                    if (backend == "CPU") {
                        binding.cpuBar.progress = progress
                    } else {
                        binding.gpuBar.progress = progress
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error processing $backend benchmark point ($x, $y)", e)
            }
        }

        // Detailed statistics logging (including all values - no outlier removal)
        if (results.isNotEmpty()) {
            val times = results.map { it.inferenceTime }
            val minTime = times.minOrNull() ?: 0L
            val maxTime = times.maxOrNull() ?: 0L
            val avgTime = times.average()
            val medianTime = times.sorted().let {
                if (it.size % 2 == 0) {
                    (it[it.size / 2 - 1] + it[it.size / 2]) / 2.0
                } else {
                    it[it.size / 2].toDouble()
                }
            }

            // Standard deviation calculation
            val variance = times.map { (it - avgTime) * (it - avgTime) }.average()
            val stdDev = sqrt(variance)

            Log.d(TAG, "📈 $backend Detailed Statistics:")
            Log.d(TAG, "   • Points processed: ${results.size}/${benchmarkCoordinates.size}")
            Log.d(TAG, "   • Min time: ${minTime}ms")
            Log.d(TAG, "   • Max time: ${maxTime}ms")
            Log.d(TAG, "   • Average: ${String.format("%.1f", avgTime)}ms")
            Log.d(TAG, "   • Median: ${String.format("%.1f", medianTime)}ms")
            Log.d(TAG, "   • Std Dev: ${String.format("%.1f", stdDev)}ms")
            Log.d(TAG, "   • Classification: Blue=${results.count { it.isBlue }}, Red=${results.count { !it.isBlue }}")

            // 🎯 Accuracy calculation and display
            val correctPredictions = results.count { it.isCorrect }
            val accuracy = (correctPredictions * 100.0 / results.size)
            Log.d(TAG, "   • Accuracy: $correctPredictions/${results.size} (${String.format("%.1f", accuracy)}%)")

            // Class-wise accuracy
            val bluePoints = results.filter { it.groundTruth }
            val redPoints = results.filter { !it.groundTruth }
            if (bluePoints.isNotEmpty()) {
                val blueAccuracy = bluePoints.count { it.isCorrect } * 100.0 / bluePoints.size
                Log.d(TAG, "     - Blue class accuracy: ${String.format("%.1f", blueAccuracy)}% (${bluePoints.count { it.isCorrect }}/${bluePoints.size})")
            }
            if (redPoints.isNotEmpty()) {
                val redAccuracy = redPoints.count { it.isCorrect } * 100.0 / redPoints.size
                Log.d(TAG, "     - Red class accuracy: ${String.format("%.1f", redAccuracy)}% (${redPoints.count { it.isCorrect }}/${redPoints.size})")
            }

            // Performance distribution analysis (including all values)
            val fastCount = times.count { it <= 1 }
            val normalCount = times.count { it in 2..5 }
            val slowCount = times.count { it > 5 }

            Log.d(TAG, "   • Performance distribution:")
            Log.d(TAG, "     - Fast (≤1ms): $fastCount points (${String.format("%.1f", fastCount * 100.0 / results.size)}%)")
            Log.d(TAG, "     - Normal (2-5ms): $normalCount points (${String.format("%.1f", normalCount * 100.0 / results.size)}%)")
            Log.d(TAG, "     - Slow (>5ms): $slowCount points (${String.format("%.1f", slowCount * 100.0 / results.size)}%)")

            // Performance consistency evaluation
            val consistencyScore = if (avgTime > 0) {
                100.0 - (stdDev / avgTime * 100.0).coerceAtMost(100.0)
            } else 0.0
            Log.d(TAG, "   • Performance consistency: ${String.format("%.1f", consistencyScore)}%")

            // Range analysis
            val range = maxTime - minTime
            Log.d(TAG, "   • Performance range: ${range}ms (${minTime}ms ~ ${maxTime}ms)")
        }

        Log.d(TAG, "🏁 $backend benchmark completed: ${results.size}/${benchmarkCoordinates.size} points")
        return results
    }

    @SuppressLint("DefaultLocale")
    private fun showBenchmarkComparison() {
        runOnUiThread {
            val cpu = cpuBenchmarkResult
            val gpu = gpuBenchmarkResult
            val npu = nnApiBenchmarkResult

            val results = listOfNotNull(cpu, gpu, npu).sortedBy { it.avgInferenceTime }
            val winner = results.firstOrNull()

            // Performance improvement calculation
            val cpuTime = cpu?.avgInferenceTime ?: 0L
            val bestTime = winner?.avgInferenceTime ?: 0L
            val speedupText = if (cpuTime > 0 && bestTime > 0 && winner?.backend != "CPU") {
                val speedup = cpuTime.toFloat() / bestTime
                String.format("%.1fx faster than CPU", speedup)
            } else "No speedup"

            binding.textView1.text = """
                    |🏆 PERFORMANCE BENCHMARK RESULTS
                    |
                    |🥇 Winner: ${winner?.backend ?: "Unknown"} 
                    |⚡ Best Time: ${winner?.avgInferenceTime ?: "N/A"}ms average
                    |🚀 Speedup: $speedupText
                    |
                    |━━━━━━━━━━━━━━━━━━━━━━━━
                    |
                    |📊 Average Inference Times:
                    |${cpu?.let { "   💻 CPU:    ${it.avgInferenceTime}ms" } ?: ""}
                    |${gpu?.let { "   🎉 GPU:    ${it.avgInferenceTime}ms" } ?: ""}
                    |${npu?.let { "   🧠 NPU:    ${it.avgInferenceTime}ms" } ?: ""}
                    |
                    |📈 Test Configuration:
                    |• Backends tested: ${results.size}
                    |• Points per backend: ${benchmarkCoordinates.size}
                    |• Cold start eliminated: ✅
                    |• Outlier filtering: ❌ (real performance)
                    |
                    |🔬 Technical Details:
                    |• Warm-up runs: Eliminate initialization overhead
                    |• High-precision timing: Nanosecond accuracy
                    |• Statistical analysis: Min/Max/Avg/StdDev
                    |• Hardware utilization: CPU/NPU acceleration
                    |
                    |💡 Interpretation:
                    |Lower latency = Better performance
                    |Check logs for detailed per-point analysis
                    |
                    |🎮 Navigation: Focus - ${currentFocusState.name}
                    |Use navigation keys to move between controls
                """.trimMargin()

            // Detailed statistics information display
            val detailsText = buildString {
                appendLine("📈 Detailed Performance Analysis:")
                appendLine("━━━━━━━━━━━━━━━━━━━━━━━━")
                appendLine()

                listOfNotNull(cpu, gpu, npu).forEach { result ->
                    val backendHistory = classificationHistory.filter {
                        // Filter points used in benchmark (recent 28 each) or include all history
                        true // Include all history or can differentiate by backend
                    }
                    val backendAccuracy = if (backendHistory.isNotEmpty()) {
                        backendHistory.count { it.isCorrect } * 100.0 / backendHistory.size
                    } else 0.0

                    val icon = when(result.backend) {
                        "CPU" -> "💻"
                        "GPU" -> "🎉"
                        "NPU" -> "🧠"
                        else -> "⚡"
                    }

                    appendLine("$icon ${result.backend} Backend Results:")
                    appendLine("  📊 Points processed: ${result.pointCount}")
                    appendLine("  🎨 Classifications: Blue=${result.blueCount}, Red=${result.redCount}")
                    appendLine("  ⚡ Average latency: ${result.avgInferenceTime}ms")
                    appendLine("  ⏱️ Total time: ${result.totalTime}ms")
                    appendLine("  🚀 Throughput: ${String.format("%.1f", result.pointCount * 1000.0 / result.totalTime)} points/sec")
                    if (backendHistory.isNotEmpty()) {
                        appendLine("  🎯 Accuracy: ${String.format("%.1f", backendAccuracy)}% (${backendHistory.count { it.isCorrect }}/${backendHistory.size})")
                    }

                    // Performance grade display
                    val grade = when {
                        result.avgInferenceTime <= 1 -> "🏆 A+ (Excellent)"
                        result.avgInferenceTime <= 3 -> "🥇 A (Very Good)"
                        result.avgInferenceTime <= 5 -> "🥈 B (Good)"
                        result.avgInferenceTime <= 10 -> "🥉 C (Fair)"
                        else -> "📉 D (Slow)"
                    }
                    appendLine("  📈 Performance Grade: $grade")
                    appendLine()
                }

                // Overall performance comparison
                if (results.size > 1) {
                    appendLine("🏁 Performance Rankings:")
                    appendLine("━━━━━━━━━━━━━━━━━━━━━━━━")
                    results.forEachIndexed { index, result ->
                        val rank = when(index) {
                            0 -> "🥇 #1"
                            1 -> "🥈 #2"
                            2 -> "🥉 #3"
                            else -> "🏃 #${index + 1}"
                        }
                        val improvement = if (index == 0) "Fastest" else {
                            val ratio = result.avgInferenceTime.toFloat() / results[0].avgInferenceTime
                            "${String.format("%.1f", ratio)}x slower"
                        }
                        appendLine("  $rank ${result.backend}: ${result.avgInferenceTime}ms ($improvement)")
                    }
                    appendLine()
                    appendLine("📋 Notes:")
                    appendLine("• Results include natural performance variation")
                    appendLine("• Warm-up eliminates cold start overhead")
                    appendLine("• Lower latency = Better performance")
                    appendLine("• NPU may show higher variation due to scheduling")
                    appendLine("• Check verbose logs for per-point timings")
                }

                appendLine()
                appendLine("🎮 Enhanced Navigation:")
                appendLine("• Current Focus: ${currentFocusState.name}")
                appendLine("• D-pad: Move cursor (coordinate area)")
                appendLine("• Navigation keys: Move between controls")
                appendLine("• Up/Down in text areas: Scroll content")
                appendLine("• 🚪 Exit key: Leave area → Focus Classify")
                appendLine("• Results area: Exit key to exit")
            }

            binding.textView2.text = detailsText

            // 📜 Scroll to top when benchmark results are displayed
            scrollToTop()
        }
    }

    fun onGPUClick(v: View) {
        Log.d(TAG, "🔘 GPU/NPU toggle clicked, current state: ${binding.gpuToggle.isChecked}")
        Log.d(TAG, "🔍 Available accelerators - GPU: $isGpuAvailable, NPU: $isNnApiAvailable")

        if (!isGpuAvailable && !isNnApiAvailable) {
            Log.w(TAG, "❌ No hardware acceleration available")
            binding.gpuToggle.isChecked = false
            return
        }

        val useAccelerator = binding.gpuToggle.isChecked
        Log.d(TAG, "🔄 Hardware acceleration toggled to: $useAccelerator")

        // Current interpreter status check
        Log.d(TAG, "🔍 Interpreter status - CPU: ${cpuInterpreter != null}, GPU: ${gpuInterpreter != null}, NPU: ${npuInterpreter != null}")

        when {
            useAccelerator && isGpuAvailable && isNnApiAvailable -> {
                Log.i(TAG, "🚀 Both GPU and NPU acceleration enabled")
            }
            useAccelerator && isGpuAvailable -> {
                Log.i(TAG, "🎉 GPU acceleration enabled")
            }
            useAccelerator && isNnApiAvailable -> {
                Log.i(TAG, "🧠 NPU acceleration enabled")
            }
            !useAccelerator -> {
                Log.i(TAG, "💻 CPU only mode enabled")
            }
            else -> {
                Log.w(TAG, "❌ No acceleration available")
            }
        }

        // Reset UI state
        binding.cpuBar.progress = 0
        binding.gpuBar.progress = 0

        Log.d(TAG, "✅ GPU toggle operation completed successfully")
    }

    private fun clearClassificationHistory() {
        classificationHistory.clear()
        runOnUiThread {
            // Restore to initial state
            binding.textView1.text = """
                    |🎯 Classification Results
                    |
                    |Classification history cleared!
                    |
                    |📱 Enhanced Navigation Controls:
                    |• Use D-pad to move the cursor (coordinate area)
                    |• Press Center button to classify current point
                    |• Use navigation keys to move between controls
                    |• 🎮 Nav button: Switch to visualization area
                    |• 🚪 Exit key: Leave area → Focus to Classify button
                    |• Results will show predicted vs actual class
                    |• ✅ = Correct prediction, ❌ = Wrong prediction
                    |
                    |🚀 Features:
                    |• Real-time neural network inference
                    |• Hardware acceleration (CPU/NPU)
                    |• Accuracy tracking and statistics
                    |• Interactive coordinate visualization
                    |• Enhanced navigation and focus management
                    |• Dual-area cursor control
                    |
                    |Ready for new classifications!
                    |
                    |🎮 Current Focus: ${currentFocusState.name}
                """.trimMargin()

            binding.textView2.text = """
                    |📊 History & Statistics
                    |
                    |No classifications yet
                    |
                    |Statistics will appear here after running
                    |classifications or benchmarks.
                    |
                    |🎯 Classification accuracy tracking
                    |📈 Performance monitoring
                    |🚀 Hardware acceleration metrics
                    |
                    |🎮 Enhanced Navigation Controls:
                    |• Focus: ${currentFocusState.name}
                    |• D-pad: Move cursor (coordinate/visualization areas)
                    |• Nav keys: Move between buttons
                    |• Up/Down in text areas: Scroll content
                    |• Center: Execute current action
                    |• 🎮 Nav button: Switch to visualization area
                    |• 🚪 Exit key: Leave area → Focus Classify
                    |• Results area: Use Exit key to exit
                """.trimMargin()

            // Restore to default colors
            binding.textView1.setTextColor(Color.BLACK)
            binding.textView2.setTextColor(Color.BLACK)

            drawCoordinateSystem()

            // 📜 Scroll to top after history clear
            scrollToTop()

            Log.d(TAG, "🧹 Classification history cleared and UI reset")
        }
    }

    private fun updateCoordinateDisplay() {
        runOnUiThread {
            // Also display ground truth information with enhanced TV visibility
            val groundTruth = getGroundTruth(currentX, currentY)
            val groundTruthClass = if (groundTruth) "Blue" else "Red"
            val focusIndicator = when (currentFocusState) {
                FocusState.COORDINATE_AREA -> "Coordinate Control"
                FocusState.BUTTON_CLASSIFY -> "Classify Button"
                FocusState.BUTTON_BENCHMARK -> "Benchmark Button"
                FocusState.BUTTON_GPU_TOGGLE -> "GPU Toggle"
                FocusState.BUTTON_NAVIGATE -> "Navigate Button"
                FocusState.RESULT_AREA_1 -> "Results Area 1"
                FocusState.RESULT_AREA_2 -> "Results Area 2"
                FocusState.VISUALIZATION_AREA -> "🎮 NEURAL VISUALIZATION" // 🔧 Make it more obvious
            }

            binding.coordinateText.text = "Position: (${String.format("%.2f", currentX)}, ${String.format("%.2f", currentY)}) | Expected: $groundTruthClass | Focus: $focusIndicator"

            // 🔧 Debug: Log coordinate display updates to track state changes
            Log.v(TAG, "📍 Coordinate display updated - Focus: $focusIndicator (${currentFocusState.name})")
        }
    }

    private fun drawCoordinateSystem() {
        val bitmap = Bitmap.createBitmap(600, 400, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // Background
        canvas.drawColor(Color.WHITE)

        val paint = Paint().apply {
            isAntiAlias = true
            strokeWidth = 2f
        }

        // Coordinate axes
        paint.color = Color.GRAY
        canvas.drawLine(300f, 0f, 300f, 400f, paint) // Y-axis
        canvas.drawLine(0f, 200f, 600f, 200f, paint) // X-axis

        // Grid
        paint.strokeWidth = 1f
        paint.color = Color.LTGRAY
        for (i in 0..6) {
            val x = i * 100f
            canvas.drawLine(x, 0f, x, 400f, paint)
        }
        for (i in 0..4) {
            val y = i * 100f
            canvas.drawLine(0f, y, 600f, y, paint)
        }

        // Draw classified points (distinguish correct/incorrect)
        for (point in classificationHistory) {
            val screenX = ((point.x - X_MIN) / (X_MAX - X_MIN) * 600).coerceIn(0f, 600f)
            val screenY = (400 - (point.y - Y_MIN) / (Y_MAX - Y_MIN) * 400).coerceIn(0f, 400f)

            // Color based on prediction result
            val predictedColor = if (point.isBlue) Color.BLUE else Color.RED

            // Visualization based on correctness
            if (point.isCorrect) {
                // Correct: filled circle
                paint.color = predictedColor
                paint.style = Paint.Style.FILL
                canvas.drawCircle(screenX, screenY, 8f, paint)
            } else {
                // Incorrect: outline circle + X mark
                paint.color = predictedColor
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = 3f
                canvas.drawCircle(screenX, screenY, 8f, paint)

                // X mark to emphasize incorrect prediction
                paint.color = Color.BLACK
                paint.strokeWidth = 2f
                canvas.drawLine(screenX - 5f, screenY - 5f, screenX + 5f, screenY + 5f, paint)
                canvas.drawLine(screenX - 5f, screenY + 5f, screenX + 5f, screenY - 5f, paint)
            }
        }

        // Current cursor position (more prominent)
        val cursorX = ((currentX - X_MIN) / (X_MAX - X_MIN) * 600).coerceIn(0f, 600f)
        val cursorY = (400 - (currentY - Y_MIN) / (Y_MAX - Y_MIN) * 400).coerceIn(0f, 400f)

        paint.color = Color.BLACK
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f
        canvas.drawCircle(cursorX, cursorY, 15f, paint)

        // Cursor center point
        paint.style = Paint.Style.FILL
        paint.strokeWidth = 2f
        canvas.drawCircle(cursorX, cursorY, 3f, paint)

        runOnUiThread {
            binding.coordinateView.setImageBitmap(bitmap)
        }
    }

    // 🔧 Safe message display (completely disable Toast)
    private fun safeShowToast(message: String) {
        Log.d(TAG, "📢 Message: $message")

        // No Toast usage to prevent SystemUI errors
        // Only use direct UI display method
        displayMessageInUI(message)
    }

    // 🎨 Display message in UI (Toast replacement) + auto-scroll - DISABLED to reduce UI clutter
    private fun displayMessageInUI(message: String) {
        // 🔧 Disabled to prevent UI clutter - only log the message
        Log.d(TAG, "📢 Message (UI display disabled): $message")

        // UI message display disabled to keep interface clean
        // All messages are still logged for debugging purposes
    }

    private fun showError(message: String) {
        Log.e(TAG, "❌ Error: $message")

        runOnUiThread {
            // Display error directly in UI instead of Toast
            displayMessageInUI("❌ Error: $message")

            binding.textView1.text = "❌ Error: $message"
            binding.textView2.text = "Please check logs for details"
            binding.textView1.setTextColor(Color.RED)
            binding.textView2.setTextColor(Color.RED)

            // Restore default color after 10 seconds
            mainHandler.postDelayed({
                binding.textView1.setTextColor(Color.BLACK)
                binding.textView2.setTextColor(Color.BLACK)
            }, 10000)
        }
    }
}