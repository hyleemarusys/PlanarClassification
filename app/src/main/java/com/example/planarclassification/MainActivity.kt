package com.example.planarclassification

import android.annotation.SuppressLint
import android.app.Activity
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.widget.Toast
import com.example.planarclassification.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.*

class MainActivity : Activity() {
    companion object {
        private const val TAG = "NeuralPlayground"

        // 원본 데이터셋 좌표 범위 (코세라 과제)
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
    }

    private lateinit var binding: ActivityMainBinding
    private var startTime: Long = 0
    private var inferenceTime: Long = 0
    private var firstFrame: Boolean = true
    private var tfliteModel: MappedByteBuffer? = null
    private var inputBuffer: ByteBuffer? = null
    private var tflite: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var isRunning = false

    // 현재 좌표 (리모컨으로 조작)
    private var currentX: Float = 0.0f
    private var currentY: Float = 0.0f
    private val moveStep = 0.2f // 리모컨 이동 간격

    // 분류 결과 저장
    private val classificationHistory = mutableListOf<ClassificationPoint>()

    data class ClassificationPoint(
        val x: Float,
        val y: Float,
        val probability: Float,
        val isBlue: Boolean,
        val inferenceTime: Long
    )

    // 벤치마크 결과 저장용
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)

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

            // 입력 버퍼 초기화
            inputBuffer = ByteBuffer.allocateDirect(4 * 2) // 2 floats
            inputBuffer?.order(ByteOrder.nativeOrder())

            Log.d(TAG, "Planar Classifier initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Error during onCreate", e)
            showError("Failed to initialize app: ${e.message}")
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
        // 버튼 클릭 리스너 설정
        binding.classifyButton.setOnClickListener { onClassifyClick(it) }
        binding.benchmarkButton.setOnClickListener { onBenchmarkClick(it) }
        binding.gpuToggle.setOnClickListener { onGPUClick(it) }
        binding.clearButton.setOnClickListener { onClearClick(it) }

        // 초기 UI 상태 설정
        binding.textView1.text = "Ready to run planar classification"
        binding.textView2.text = "Use D-pad to move cursor, Center to classify"
        binding.cpuBar.progress = 0
        binding.gpuBar.progress = 0

        // 텍스트 색상을 검정색으로 설정
        binding.textView1.setTextColor(Color.BLACK)
        binding.textView2.setTextColor(Color.BLACK)

        updateCoordinateDisplay()
    }

    private fun setupVisualization() {
        // 좌표계 캔버스 초기화
        drawCoordinateSystem()
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanupResources()
    }

    private fun cleanupResources() {
        try {
            tflite?.close()
            tflite = null
            gpuDelegate?.close()
            gpuDelegate = null
            inputBuffer = null
            tfliteModel = null
            cpuBenchmarkResult = null
            gpuBenchmarkResult = null
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
                if (!isRunning) {
                    classifyCurrentPoint()
                }
                return true
            }
            KeyEvent.KEYCODE_MENU -> {
                if (!isRunning) {
                    runBenchmark()
                }
                return true
            }
            KeyEvent.KEYCODE_BACK -> {
                clearClassificationHistory()
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    fun onClassifyClick(v: View) {
        if (isRunning) {
            Toast.makeText(this, "Already running classification...", Toast.LENGTH_SHORT).show()
            return
        }
        classifyCurrentPoint()
    }

    fun onBenchmarkClick(v: View) {
        if (isRunning) {
            Toast.makeText(this, "Already running benchmark...", Toast.LENGTH_SHORT).show()
            return
        }
        runBenchmark()
    }

    fun onClearClick(v: View) {
        clearClassificationHistory()
    }

    private fun classifyCurrentPoint() {
        if (tfliteModel == null) {
            showError("TensorFlow Lite model not loaded")
            return
        }

        thread {
            isRunning = true
            try {
                val useGPU = binding.gpuToggle.isChecked
                val result = runSingleClassification(currentX, currentY, useGPU)

                result?.let { point ->
                    classificationHistory.add(point)

                    runOnUiThread {
                        val className = if (point.isBlue) "Blue" else "Red"
                        val confidence = if (point.isBlue) point.probability else (1f - point.probability)

                        binding.textView1.text = "Classification Result:\n" +
                                "Point: (${String.format("%.2f", point.x)}, ${String.format("%.2f", point.y)})\n" +
                                "Class: $className (${String.format("%.1f", confidence * 100)}%)\n" +
                                "Inference time: ${point.inferenceTime}ms"

                        binding.textView2.text = "History: ${classificationHistory.size} points classified\n" +
                                "Blue: ${classificationHistory.count { it.isBlue }}, " +
                                "Red: ${classificationHistory.count { !it.isBlue }}\n" +
                                "Backend: ${if (useGPU) "GPU" else "CPU"}"

                        drawCoordinateSystem()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error during classification", e)
                runOnUiThread {
                    showError("Classification failed: ${e.message}")
                }
            } finally {
                isRunning = false
            }
        }
    }

    private fun runBenchmark() {
        thread {
            isRunning = true
            try {
                runOnUiThread {
                    binding.textView1.text = "Running benchmark..."
                    binding.textView2.text = "Testing CPU performance..."
                    binding.cpuBar.progress = 0
                    binding.gpuBar.progress = 0
                }

                // CPU 벤치마크
                Log.d(TAG, "Starting CPU benchmark")
                val cpuStartTime = SystemClock.uptimeMillis()
                val cpuResults = runBenchmarkForBackend(false)
                val cpuTotalTime = SystemClock.uptimeMillis() - cpuStartTime

                cpuBenchmarkResult = BenchmarkResult(
                    avgInferenceTime = cpuResults.map { it.inferenceTime }.average().toLong(),
                    totalTime = cpuTotalTime,
                    backend = "CPU",
                    pointCount = cpuResults.size,
                    blueCount = cpuResults.count { it.isBlue },
                    redCount = cpuResults.count { !it.isBlue }
                )

                runOnUiThread {
                    binding.textView1.text = "CPU benchmark completed!\nStarting GPU benchmark..."
                    binding.textView2.text = "Testing GPU performance..."
                }

                Thread.sleep(1000)

                // GPU 벤치마크
                Log.d(TAG, "Starting GPU benchmark")
                val gpuStartTime = SystemClock.uptimeMillis()
                val gpuResults = runBenchmarkForBackend(true)
                val gpuTotalTime = SystemClock.uptimeMillis() - gpuStartTime

                gpuBenchmarkResult = BenchmarkResult(
                    avgInferenceTime = gpuResults.map { it.inferenceTime }.average().toLong(),
                    totalTime = gpuTotalTime,
                    backend = "GPU",
                    pointCount = gpuResults.size,
                    blueCount = gpuResults.count { it.isBlue },
                    redCount = gpuResults.count { !it.isBlue }
                )

                // 결과 비교 표시
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

    private fun runBenchmarkForBackend(useGPU: Boolean): List<ClassificationPoint> {
        val results = mutableListOf<ClassificationPoint>()

        val options = Interpreter.Options()
        if (useGPU) {
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.d(TAG, "GPU Delegate applied for benchmark")
            } catch (e: Exception) {
                Log.w(TAG, "GPU not available for benchmark, using CPU", e)
                options.setNumThreads(4)
            }
        } else {
            options.setNumThreads(4)
        }

        val interpreter = Interpreter(tfliteModel!!, options)

        benchmarkCoordinates.forEachIndexed { index, (x, y) ->
            try {
                val point = classifyPoint(interpreter, x, y)
                if (point != null) {
                    results.add(point)
                }

                val progress = ((index + 1) * 100) / benchmarkCoordinates.size
                runOnUiThread {
                    if (useGPU) {
                        binding.gpuBar.progress = progress
                    } else {
                        binding.cpuBar.progress = progress
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error processing benchmark point ($x, $y)", e)
            }
        }

        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null

        return results
    }

    private fun runSingleClassification(x: Float, y: Float, useGPU: Boolean): ClassificationPoint? {
        val options = Interpreter.Options()
        if (useGPU) {
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            } catch (e: Exception) {
                Log.w(TAG, "GPU not available, using CPU", e)
                runOnUiThread {
                    binding.gpuToggle.isChecked = false
                }
                options.setNumThreads(4)
            }
        } else {
            options.setNumThreads(4)
        }

        val interpreter = Interpreter(tfliteModel!!, options)
        val result = classifyPoint(interpreter, x, y)

        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null

        return result
    }

    private fun classifyPoint(interpreter: Interpreter, x: Float, y: Float): ClassificationPoint? {
        try {
            // 입력 준비
            inputBuffer?.rewind()
            inputBuffer?.putFloat(x)
            inputBuffer?.putFloat(y)

            // 출력 준비
            val outputBuffer = ByteBuffer.allocateDirect(4) // 1 float
            outputBuffer.order(ByteOrder.nativeOrder())

            // 추론 실행
            val startTime = SystemClock.uptimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = SystemClock.uptimeMillis() - startTime

            // 결과 추출
            outputBuffer.rewind()
            val probability = outputBuffer.float
            val isBlue = probability >= 0.5f

            return ClassificationPoint(x, y, probability, isBlue, inferenceTime)

        } catch (e: Exception) {
            Log.e(TAG, "Error classifying point ($x, $y)", e)
            return null
        }
    }

    @SuppressLint("DefaultLocale", "SetTextI18n")
    private fun showBenchmarkComparison() {
        runOnUiThread {
            val cpu = cpuBenchmarkResult
            val gpu = gpuBenchmarkResult

            if (cpu != null && gpu != null) {
                val speedup = cpu.avgInferenceTime.toFloat() / gpu.avgInferenceTime.toFloat()
                val winner = if (cpu.avgInferenceTime < gpu.avgInferenceTime) "CPU" else "GPU"

                binding.textView1.text = """
                    |BENCHMARK RESULTS
                    |
                    |🏆 Winner: $winner
                    |📊 Performance Comparison:
                    |   CPU:    ${cpu.avgInferenceTime}ms avg
                    |   GPU:    ${gpu.avgInferenceTime}ms avg
                    |
                    |⚡ Speedup: ${String.format("%.2fx", if (speedup > 1) speedup else 1/speedup)}
                """.trimMargin()

                binding.textView2.text = """
                    |📈 Classification Results:
                    |
                    |CPU Backend (${cpu.pointCount} points):
                    |  • Blue: ${cpu.blueCount}, Red: ${cpu.redCount}
                    |  • Avg inference: ${cpu.avgInferenceTime}ms
                    |  • Total time: ${cpu.totalTime}ms
                    |
                    |GPU Backend (${gpu.pointCount} points):
                    |  • Blue: ${gpu.blueCount}, Red: ${gpu.redCount}
                    |  • Avg inference: ${gpu.avgInferenceTime}ms
                    |  • Total time: ${gpu.totalTime}ms
                """.trimMargin()

                Log.d(TAG, "Benchmark completed - CPU: ${cpu.avgInferenceTime}ms, GPU: ${gpu.avgInferenceTime}ms")
            }
        }
    }

    fun onGPUClick(v: View) {
        val useGPU = binding.gpuToggle.isChecked
        val message = if (useGPU) {
            "GPU acceleration will be used for inference"
        } else {
            "GPU is disabled; CPU will be used"
        }
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

        binding.cpuBar.progress = 0
        binding.gpuBar.progress = 0
    }

    private fun clearClassificationHistory() {
        classificationHistory.clear()
        runOnUiThread {
            binding.textView1.text = "Classification history cleared"
            binding.textView2.text = "Use D-pad to move cursor, Center to classify"
            drawCoordinateSystem()
        }
    }

    private fun updateCoordinateDisplay() {
        runOnUiThread {
            binding.coordinateText.text = "Current: (${String.format("%.2f", currentX)}, ${String.format("%.2f", currentY)})"
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

        // 분류된 점들 그리기
        for (point in classificationHistory) {
            val screenX = ((point.x - X_MIN) / (X_MAX - X_MIN) * 600).coerceIn(0f, 600f)
            val screenY = (400 - (point.y - Y_MIN) / (Y_MAX - Y_MIN) * 400).coerceIn(0f, 400f)

            paint.color = if (point.isBlue) Color.BLUE else Color.RED
            paint.style = Paint.Style.FILL
            canvas.drawCircle(screenX, screenY, 8f, paint)
        }

        // 현재 커서 위치
        val cursorX = ((currentX - X_MIN) / (X_MAX - X_MIN) * 600).coerceIn(0f, 600f)
        val cursorY = (400 - (currentY - Y_MIN) / (Y_MAX - Y_MIN) * 400).coerceIn(0f, 400f)

        paint.color = Color.BLACK
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f
        canvas.drawCircle(cursorX, cursorY, 12f, paint)

        runOnUiThread {
            binding.coordinateView.setImageBitmap(bitmap)
        }
    }

    private fun showError(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            binding.textView1.text = "Error: $message"
            binding.textView2.text = ""
            binding.textView1.setTextColor(Color.BLACK)
            binding.textView2.setTextColor(Color.BLACK)
        }
        Log.e(TAG, message)
    }
}