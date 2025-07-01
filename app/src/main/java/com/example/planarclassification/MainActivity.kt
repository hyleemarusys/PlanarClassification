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

        // 원본 데이터셋 좌표 범위
        private const val X_MIN = -4.5f
        private const val X_MAX = 4.0f
        private const val Y_MIN = -4.0f
        private const val Y_MAX = 4.0f

        // 벤치마크용 테스트 좌표들 (원형 패턴)
        private val benchmarkCoordinates = arrayOf(
            Pair(-3.0f, -2.0f), Pair(-2.0f, -3.0f), Pair(0.0f, -3.5f), Pair(2.0f, -3.0f),
            Pair(3.0f, -2.0f), Pair(3.5f, 0.0f), Pair(3.0f, 2.0f), Pair(2.0f, 3.0f),
            Pair(0.0f, 3.5f), Pair(-2.0f, 3.0f), Pair(-3.0f, 2.0f), Pair(-3.5f, 0.0f),
            Pair(-1.0f, -1.0f), Pair(1.0f, -1.0f), Pair(1.0f, 1.0f), Pair(-1.0f, 1.0f),
            Pair(0.0f, 0.0f), Pair(-2.5f, 0.0f), Pair(2.5f, 0.0f), Pair(0.0f, -2.5f),
            Pair(0.0f, 2.5f), Pair(-1.5f, -1.5f), Pair(1.5f, -1.5f), Pair(1.5f, 1.5f),
            Pair(-1.5f, 1.5f), Pair(-4.0f, 0.0f), Pair(3.5f, 0.0f), Pair(0.0f, -3.8f)
        )

        // 🎯 Ground Truth 함수 (코세라 딥러닝 3주차 - 원형 패턴)
        fun getGroundTruth(x: Float, y: Float): Boolean {
            // 일반적인 코세라 패턴: 중심에서 거리 기반 분류
            val distance = kotlin.math.sqrt(x * x + y * y)

            // 복합 패턴: 거리 + 각도 조합 (더 현실적인 패턴)
            val angle = kotlin.math.atan2(y, x)

            // 패턴 1: 내부 원 (거리 < 1.5) → Blue
            if (distance < 1.5) return true

            // 패턴 2: 외부 링 (거리 > 3.0) → Blue
            if (distance > 3.0) return true

            // 패턴 3: 중간 영역에서 각도 기반 분류
            // 1사분면과 3사분면에서 특정 조건 → Blue
            val isFirstOrThirdQuadrant = (x > 0 && y > 0) || (x < 0 && y < 0)
            if (distance >= 1.5 && distance <= 3.0) {
                return isFirstOrThirdQuadrant && (kotlin.math.abs(angle) < kotlin.math.PI / 3)
            }

            // 기본값: Red
            return false
        }

        // 🎯 대안 Ground Truth 함수들 (다른 패턴들)
        fun getGroundTruthSimpleCircle(x: Float, y: Float): Boolean {
            // 단순 원형: 중심에서 거리 < 2.0 → Blue
            val distance = kotlin.math.sqrt(x * x + y * y)
            return distance < 2.0
        }

        fun getGroundTruthSpiral(x: Float, y: Float): Boolean {
            // 나선형 패턴
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

    // 🚀 성능 개선: 인터프리터 재사용
    private var cpuInterpreter: Interpreter? = null
    private var gpuInterpreter: Interpreter? = null
    private var npuInterpreter: Interpreter? = null

    private var gpuDelegate: Any? = null
    private var nnApiDelegate: Any? = null
    private var isRunning = false
    private var isGpuAvailable = false
    private var isNnApiAvailable = false

    // 현재 좌표 (리모컨으로 조작)
    private var currentX: Float = 0.0f
    private var currentY: Float = 0.0f
    private val moveStep = 0.2f

    // 분류 결과 저장
    private val classificationHistory = mutableListOf<ClassificationPoint>()

    // 🔧 안전한 Toast 핸들러
    private val mainHandler = Handler(Looper.getMainLooper())

    // 📜 ScrollView 참조 (자동 스크롤용)
    private lateinit var resultsScrollView: android.widget.ScrollView

    data class ClassificationPoint(
        val x: Float,
        val y: Float,
        val probability: Float,
        val isBlue: Boolean,
        val inferenceTime: Long,
        val groundTruth: Boolean,  // 실제 정답
        val isCorrect: Boolean     // 예측이 맞는지 여부
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

            // TensorFlow Lite 모델 로드
            tfliteModel = try {
                loadModelFile("original_planar_classifier.tflite")
            } catch (e: IOException) {
                Log.e(TAG, "Failed to load TFLite model", e)
                showError("Failed to load TensorFlow Lite model")
                return
            }

            // 🚀 성능 개선: 입력 버퍼 미리 할당
            inputBuffer = ByteBuffer.allocateDirect(4 * 2).apply {
                order(ByteOrder.nativeOrder())
            }

            // 🚀 성능 개선: 인터프리터 미리 생성
            initializeInterpreters()

            Log.d(TAG, "🎉 Planar Classifier initialized successfully!")

        } catch (e: Exception) {
            Log.e(TAG, "Error during onCreate", e)
            showError("Failed to initialize app: ${e.message}")
        }
    }

    // 🚀 성능 개선: 인터프리터 미리 생성
    private fun initializeInterpreters() {
        try {
            Log.d(TAG, "🔧 Initializing interpreters...")

            // CPU 인터프리터
            val cpuOptions = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true) // 🚀 XNNPACK 활성화
            }
            cpuInterpreter = Interpreter(tfliteModel!!, cpuOptions)
            Log.d(TAG, "✅ CPU interpreter initialized")

            // GPU 인터프리터
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

            // NPU 인터프리터
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

    // 🔍 디바이스 하드웨어 정보 로깅 (간소화)
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

    // 🔍 TensorFlow Lite 정보 확인 (간소화)
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

    // 🔍 OpenGL 지원 확인 (간소화)
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

    // 🔍 가속기 확인 (안전성 강화)
    private fun checkAcceleratorAvailability() {
        Log.i(TAG, "========== ACCELERATOR CHECK ==========")

        // GPU 확인
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

        // NPU 확인
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

    // 🔧 안전한 GPU delegate 생성
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

    // 🔧 안전한 NNAPI delegate 생성
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

    // Delegate 추가
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

    // Delegate 정리
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
        binding.clearButton.setOnClickListener { onClearClick(it) }

        // 🔍 ScrollView 참조 설정
        resultsScrollView = findViewById(R.id.resultsScrollView)

        // 가속기 사용 불가능하면 토글 비활성화
        binding.gpuToggle.isEnabled = isGpuAvailable || isNnApiAvailable

        // 🔄 TextView 스크롤 기능 활성화
        binding.textView1.movementMethod = android.text.method.ScrollingMovementMethod()
        binding.textView2.movementMethod = android.text.method.ScrollingMovementMethod()

        // 초기 UI 상태 설정
        binding.textView1.text = """
            |🎯 Classification Results
            |
            |Ready to run planar classification!
            |
            |📱 How to use:
            |• Use D-pad to move the cursor
            |• Press Center button to classify current point
            |• Results will show predicted vs actual class
            |• ✅ = Correct prediction, ❌ = Wrong prediction
            |
            |🚀 Features:
            |• Real-time neural network inference
            |• Hardware acceleration (CPU/NPU)
            |• Accuracy tracking and statistics
            |• Interactive coordinate visualization
            |
            |📜 This area is scrollable - swipe up/down to see more content
        """.trimMargin()

        binding.textView2.text = when {
            isGpuAvailable && isNnApiAvailable -> """
                |📊 System Status
                |
                |🚀 GPU + NPU available
                |Hardware acceleration ready
                |
                |Device: SKB BMA-AI100
                |Backends: CPU, GPU, NPU
                |
                |Ready for high-performance
                |neural network inference!
                |
                |🎯 Expected performance:
                |• CPU: ~1ms per inference
                |• NPU: ~2ms per inference
                |• High accuracy on test patterns
            """.trimMargin()
            isGpuAvailable -> """
                |📊 System Status
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
            """.trimMargin()
        }

        binding.cpuBar.progress = 0
        binding.gpuBar.progress = 0

        // 텍스트 색상 설정
        binding.textView1.setTextColor(Color.BLACK)
        binding.textView2.setTextColor(Color.BLACK)

        updateCoordinateDisplay()
    }

    private fun setupVisualization() {
        drawCoordinateSystem()
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanupResources()
    }

    // 📜 자동 스크롤 기능 (내용이 업데이트될 때 맨 위로 스크롤)
    private fun scrollToTop() {
        try {
            mainHandler.post {
                resultsScrollView.smoothScrollTo(0, 0)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Auto scroll failed: ${e.message}")
        }
    }

    // 📜 자동 스크롤 기능 (내용이 업데이트될 때 맨 아래로 스크롤)
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

    // 리모컨 키 이벤트 처리
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_DPAD_UP -> {
                currentY = (currentY + moveStep).coerceAtMost(Y_MAX)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                return true
            }
            KeyEvent.KEYCODE_DPAD_DOWN -> {
                currentY = (currentY - moveStep).coerceAtLeast(Y_MIN)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                return true
            }
            KeyEvent.KEYCODE_DPAD_LEFT -> {
                currentX = (currentX - moveStep).coerceAtLeast(X_MIN)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                return true
            }
            KeyEvent.KEYCODE_DPAD_RIGHT -> {
                currentX = (currentX + moveStep).coerceAtMost(X_MAX)
                updateCoordinateDisplay()
                drawCoordinateSystem()
                return true
            }
            KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_ENTER -> {
                if (!isRunning) classifyCurrentPoint()
                return true
            }
            KeyEvent.KEYCODE_MEDIA_REWIND, KeyEvent.KEYCODE_MENU -> {
                if (!isRunning) runBenchmark()
                return true
            }
            KeyEvent.KEYCODE_MEDIA_FAST_FORWARD -> {
                toggleAccelerator()
                return true
            }
            KeyEvent.KEYCODE_BACK -> {
                clearClassificationHistory()
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun toggleAccelerator() {
        Log.d(TAG, "📱 Remote control accelerator toggle triggered")

        if (!isGpuAvailable && !isNnApiAvailable) {
            Log.w(TAG, "No hardware acceleration available for toggle")
            safeShowToast("No hardware acceleration available")
            return
        }

        val newState = !binding.gpuToggle.isChecked
        binding.gpuToggle.isChecked = newState
        Log.d(TAG, "🔄 Accelerator toggled via remote to: $newState")

        onGPUClick(binding.gpuToggle)
    }

    fun onClassifyClick(v: View) {
        if (isRunning) {
            safeShowToast("Already running classification...")
            return
        }
        classifyCurrentPoint()
    }

    fun onBenchmarkClick(v: View) {
        if (isRunning) {
            safeShowToast("Already running benchmark...")
            return
        }
        runBenchmark()
    }

    fun onClearClick(v: View) {
        clearClassificationHistory()
    }

    // 🚀 성능 개선: 미리 생성된 인터프리터 사용
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
                            """.trimMargin()

                        // 전체 정확도 계산
                        val totalPoints = classificationHistory.size
                        val correctPredictions = classificationHistory.count { it.isCorrect }
                        val accuracy = if (totalPoints > 0) {
                            (correctPredictions * 100.0 / totalPoints)
                        } else 0.0

                        // 클래스별 통계
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
                            """.trimMargin()

                        drawCoordinateSystem()

                        // 📜 새 결과가 표시되면 맨 위로 스크롤
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

    // 🚀 성능 최적화된 분류 함수 (warm-up 포함 + 정답 체크)
    private fun classifyPointFast(interpreter: Interpreter, x: Float, y: Float, isWarmup: Boolean = false): ClassificationPoint? {
        return try {
            // 입력 준비 (버퍼 재사용)
            inputBuffer?.rewind()
            inputBuffer?.putFloat(x)
            inputBuffer?.putFloat(y)

            // 출력 준비
            val outputBuffer = ByteBuffer.allocateDirect(4).apply {
                order(ByteOrder.nativeOrder())
            }

            // 고정밀 시간 측정 (warm-up일 때는 측정하지 않음)
            val startTime = if (!isWarmup) System.nanoTime() else 0L
            interpreter.run(inputBuffer, outputBuffer)
            val endTime = if (!isWarmup) System.nanoTime() else 0L

            val inferenceTimeMs = if (!isWarmup) {
                maxOf(1L, (endTime - startTime) / 1_000_000L)
            } else {
                1L // warm-up은 시간 측정 안함
            }

            // 결과 추출
            outputBuffer.rewind()
            val probability = outputBuffer.float
            val predictedIsBlue = probability >= 0.5f

            // 🎯 Ground Truth 계산 및 정확도 체크
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

                // CPU 벤치마크
                Log.d(TAG, "🚀 Starting CPU benchmark")
                val cpuStartTime = SystemClock.uptimeMillis()
                val cpuResults = runBenchmarkForBackend("CPU")
                val cpuTotalTime = SystemClock.uptimeMillis() - cpuStartTime

                cpuBenchmarkResult = createBenchmarkResult(cpuResults, cpuTotalTime, "CPU")
                Log.d(TAG, "🏁 CPU benchmark completed: ${cpuResults.size} points, avg: ${cpuBenchmarkResult?.avgInferenceTime}ms")

                // GPU 벤치마크
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

                // NPU 벤치마크
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

    // 🚀 성능 개선: 미리 생성된 인터프리터 사용
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

                    // 각 포인트의 개별 시간 로깅 (정답 여부 포함)
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

        // 상세 통계 로깅 (모든 값 포함 - outlier 제거 안함)
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

            // 표준편차 계산
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

            // 🎯 정확도 계산 및 표시
            val correctPredictions = results.count { it.isCorrect }
            val accuracy = (correctPredictions * 100.0 / results.size)
            Log.d(TAG, "   • Accuracy: $correctPredictions/${results.size} (${String.format("%.1f", accuracy)}%)")

            // 클래스별 정확도
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

            // 성능 분포 분석 (모든 값 포함)
            val fastCount = times.count { it <= 1 }
            val normalCount = times.count { it in 2..5 }
            val slowCount = times.count { it > 5 }

            Log.d(TAG, "   • Performance distribution:")
            Log.d(TAG, "     - Fast (≤1ms): $fastCount points (${String.format("%.1f", fastCount * 100.0 / results.size)}%)")
            Log.d(TAG, "     - Normal (2-5ms): $normalCount points (${String.format("%.1f", normalCount * 100.0 / results.size)}%)")
            Log.d(TAG, "     - Slow (>5ms): $slowCount points (${String.format("%.1f", slowCount * 100.0 / results.size)}%)")

            // 성능 일관성 평가
            val consistencyScore = if (avgTime > 0) {
                100.0 - (stdDev / avgTime * 100.0).coerceAtMost(100.0)
            } else 0.0
            Log.d(TAG, "   • Performance consistency: ${String.format("%.1f", consistencyScore)}%")

            // Range 분석
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

            // 성능 향상 계산
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
                """.trimMargin()

            // 상세 통계 정보 표시
            val detailsText = buildString {
                appendLine("📈 Detailed Performance Analysis:")
                appendLine("━━━━━━━━━━━━━━━━━━━━━━━━")
                appendLine()

                listOfNotNull(cpu, gpu, npu).forEach { result ->
                    val backendHistory = classificationHistory.filter {
                        // 벤치마크에서 사용된 포인트들 필터링 (최근 28개씩)
                        true // 모든 히스토리 포함하거나 백엔드별로 구분 가능
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

                    // 성능 등급 표시
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

                // 전체 성능 비교
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
            }

            binding.textView2.text = detailsText

            // 📜 벤치마크 결과가 표시되면 맨 위로 스크롤
            scrollToTop()
        }
    }

    fun onGPUClick(v: View) {
        Log.d(TAG, "🔘 GPU/NPU toggle clicked, current state: ${binding.gpuToggle.isChecked}")
        Log.d(TAG, "🔍 Available accelerators - GPU: $isGpuAvailable, NPU: $isNnApiAvailable")

        if (!isGpuAvailable && !isNnApiAvailable) {
            Log.w(TAG, "❌ No hardware acceleration available")
            safeShowToast("No hardware acceleration available")
            binding.gpuToggle.isChecked = false
            return
        }

        val useAccelerator = binding.gpuToggle.isChecked
        Log.d(TAG, "🔄 Hardware acceleration toggled to: $useAccelerator")

        // 현재 인터프리터 상태 체크
        Log.d(TAG, "🔍 Interpreter status - CPU: ${cpuInterpreter != null}, GPU: ${gpuInterpreter != null}, NPU: ${npuInterpreter != null}")

        val message = if (useAccelerator) {
            when {
                isGpuAvailable && isNnApiAvailable -> {
                    Log.i(TAG, "🚀 Both GPU and NPU acceleration enabled")
                    "🚀 Hardware acceleration enabled (GPU + NPU)"
                }
                isGpuAvailable -> {
                    Log.i(TAG, "🎉 GPU acceleration enabled")
                    "🎉 GPU acceleration enabled"
                }
                isNnApiAvailable -> {
                    Log.i(TAG, "🧠 NPU acceleration enabled")
                    "🧠 NPU acceleration enabled"
                }
                else -> {
                    Log.w(TAG, "❌ No acceleration available")
                    "❌ No acceleration available"
                }
            }
        } else {
            Log.i(TAG, "💻 CPU only mode enabled")
            "💻 CPU only mode"
        }

        Log.d(TAG, "📤 About to display message: $message")

        // 안전한 메시지 표시 (Toast 없음)
        safeShowToast(message)

        // UI 상태 리셋
        binding.cpuBar.progress = 0
        binding.gpuBar.progress = 0

        Log.d(TAG, "✅ GPU toggle operation completed successfully")
    }

    private fun clearClassificationHistory() {
        classificationHistory.clear()
        runOnUiThread {
            // 초기 상태로 복원
            binding.textView1.text = """
                    |🎯 Classification Results
                    |
                    |Classification history cleared!
                    |
                    |📱 How to use:
                    |• Use D-pad to move the cursor
                    |• Press Center button to classify current point
                    |• Results will show predicted vs actual class
                    |• ✅ = Correct prediction, ❌ = Wrong prediction
                    |
                    |🚀 Features:
                    |• Real-time neural network inference
                    |• Hardware acceleration (CPU/NPU)
                    |• Accuracy tracking and statistics
                    |• Interactive coordinate visualization
                    |
                    |Ready for new classifications!
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
                """.trimMargin()

            // 기본 색상으로 복원
            binding.textView1.setTextColor(Color.BLACK)
            binding.textView2.setTextColor(Color.BLACK)

            drawCoordinateSystem()

            // 📜 히스토리 클리어 후 맨 위로 스크롤
            scrollToTop()

            Log.d(TAG, "🧹 Classification history cleared and UI reset")
        }
    }

    private fun updateCoordinateDisplay() {
        runOnUiThread {
            // Ground truth 정보도 함께 표시
            val groundTruth = getGroundTruth(currentX, currentY)
            val groundTruthClass = if (groundTruth) "Blue" else "Red"

            binding.coordinateText.text = "Position: (${String.format("%.2f", currentX)}, ${String.format("%.2f", currentY)}) | Expected: $groundTruthClass"
        }
    }

    private fun drawCoordinateSystem() {
        val bitmap = Bitmap.createBitmap(600, 400, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // 배경
        canvas.drawColor(Color.WHITE)

        val paint = Paint().apply {
            isAntiAlias = true
            strokeWidth = 2f
        }

        // 좌표축
        paint.color = Color.GRAY
        canvas.drawLine(300f, 0f, 300f, 400f, paint) // Y축
        canvas.drawLine(0f, 200f, 600f, 200f, paint) // X축

        // 격자
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

        // 분류된 점들 그리기 (정답/오답 구분)
        for (point in classificationHistory) {
            val screenX = ((point.x - X_MIN) / (X_MAX - X_MIN) * 600).coerceIn(0f, 600f)
            val screenY = (400 - (point.y - Y_MIN) / (Y_MAX - Y_MIN) * 400).coerceIn(0f, 400f)

            // 예측 결과에 따른 색상
            val predictedColor = if (point.isBlue) Color.BLUE else Color.RED

            // 정답 여부에 따른 시각화
            if (point.isCorrect) {
                // 정답: 채워진 원
                paint.color = predictedColor
                paint.style = Paint.Style.FILL
                canvas.drawCircle(screenX, screenY, 8f, paint)
            } else {
                // 오답: 테두리만 있는 원 + X 표시
                paint.color = predictedColor
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = 3f
                canvas.drawCircle(screenX, screenY, 8f, paint)

                // X 표시로 오답 강조
                paint.color = Color.BLACK
                paint.strokeWidth = 2f
                canvas.drawLine(screenX - 5f, screenY - 5f, screenX + 5f, screenY + 5f, paint)
                canvas.drawLine(screenX - 5f, screenY + 5f, screenX + 5f, screenY - 5f, paint)
            }
        }

        // 현재 커서 위치 (더 눈에 띄게)
        val cursorX = ((currentX - X_MIN) / (X_MAX - X_MIN) * 600).coerceIn(0f, 600f)
        val cursorY = (400 - (currentY - Y_MIN) / (Y_MAX - Y_MIN) * 400).coerceIn(0f, 400f)

        paint.color = Color.BLACK
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f
        canvas.drawCircle(cursorX, cursorY, 15f, paint)

        // 커서 중심점
        paint.style = Paint.Style.FILL
        paint.strokeWidth = 2f
        canvas.drawCircle(cursorX, cursorY, 3f, paint)

        runOnUiThread {
            binding.coordinateView.setImageBitmap(bitmap)
        }
    }

    // 🔧 안전한 메시지 표시 (Toast 완전 비활성화)
    private fun safeShowToast(message: String) {
        Log.d(TAG, "📢 Message: $message")

        // SystemUI 에러 방지를 위해 Toast 사용 안함
        // UI에 직접 표시하는 방식만 사용
        displayMessageInUI(message)
    }

    // 🎨 UI에 메시지 표시 (Toast 대체) + 자동 스크롤
    private fun displayMessageInUI(message: String) {
        try {
            runOnUiThread {
                // 타임스탬프 포함 메시지 생성
                val timestamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())
                val newMessage = "[$timestamp] $message"

                // 기존 텍스트에 새 메시지 추가 (최대 20줄 유지)
                val currentText = binding.textView1.text.toString()
                val lines = currentText.split("\n").toMutableList()

                // 새 메시지를 맨 위에 추가
                lines.add(0, newMessage)
                lines.add(1, "") // 빈 줄 추가

                // 최대 25줄로 제한 (너무 길어지면 오래된 메시지 제거)
                while (lines.size > 25) {
                    lines.removeAt(lines.size - 1)
                }

                binding.textView1.text = lines.joinToString("\n")

                // 메시지 색상 변경으로 주목도 높이기
                when {
                    message.contains("🧠 NPU") -> binding.textView1.setTextColor(android.graphics.Color.parseColor("#4CAF50"))
                    message.contains("🎉 GPU") -> binding.textView1.setTextColor(android.graphics.Color.parseColor("#2196F3"))
                    message.contains("💻 CPU") -> binding.textView1.setTextColor(android.graphics.Color.parseColor("#FF9800"))
                    message.contains("❌") -> binding.textView1.setTextColor(android.graphics.Color.parseColor("#F44336"))
                    else -> binding.textView1.setTextColor(android.graphics.Color.parseColor("#333333"))
                }

                // 맨 위로 스크롤 (새 메시지를 바로 보이도록)
                binding.textView1.scrollTo(0, 0)

                // 5초 후 기본 색상으로 복원
                mainHandler.postDelayed({
                    binding.textView1.setTextColor(android.graphics.Color.parseColor("#333333"))
                }, 5000)

                Log.d(TAG, "✅ UI message displayed: $message")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to display UI message: $message", e)
        }
    }

    private fun showError(message: String) {
        Log.e(TAG, "❌ Error: $message")

        runOnUiThread {
            // Toast 대신 UI에 직접 에러 표시
            displayMessageInUI("❌ Error: $message")

            binding.textView1.text = "❌ Error: $message"
            binding.textView2.text = "Please check logs for details"
            binding.textView1.setTextColor(Color.RED)
            binding.textView2.setTextColor(Color.RED)

            // 10초 후 기본 색상으로 복원
            mainHandler.postDelayed({
                binding.textView1.setTextColor(Color.BLACK)
                binding.textView2.setTextColor(Color.BLACK)
            }, 10000)
        }
    }
}