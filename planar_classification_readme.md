# Planar Classification

Android set-top box app for real-time 2D neural network classification with remote control navigation.

## Features

- Real-time binary classification (Blue/Red) on 2D coordinates
- Hardware acceleration: CPU/NPU backends
- Performance benchmarking and accuracy tracking
- Remote control navigation for Android set-top boxes
- Interactive coordinate visualization

## Requirements

- Android 5.0+ (API 21)
- Android set-top box with remote control
- TensorFlow Lite model file

## Installation

```bash
git clone https://github.com/hyleemarusys/PlanarClassification.git
cd PlanarClassification
./gradlew assembleDebug
./gradlew installDebug
```

## Usage

### Remote Controls
- **D-pad**: Move cursor in coordinate area
- **Center**: Execute action (classify point)
- **Navigation keys**: Move between buttons (Classify/Benchmark/NPU Toggle/Navigate)
- **Menu key**: Run benchmark
- **Exit key**: Leave current area
- **Back key**: Clear classification history

## Tech Stack

- **Language**: Kotlin
- **Platform**: Android
- **ML**: TensorFlow Lite
- **Acceleration**: CPU/NPU backends

## Project Structure

```
PlanarClassification/
├── app/
│   ├── manifests/
│   │   └── AndroidManifest.xml
│   ├── kotlin+java/
│   │   └── com.example.planarclassification/
│   │       └── MainActivity.kt
│   ├── assets/
│   │   └── original_planar_classifier.tflite
│   └── res/
│       ├── layout/
│       │   └── activity_main.xml
│       ├── mipmap/
│       │   └── ic_launcher.webp (multiple versions)
│       └── values/
│           ├── colors.xml
│           ├── strings.xml
│           └── themes.xml
├── build.gradle.kts (Project)
├── build.gradle.kts (Module :app)
├── gradle.properties
└── settings.gradle.kts
```

## Model Details

- **Input**: 2D coordinates (x, y)
- **Output**: Binary probability (≥0.5 = Blue, <0.5 = Red)
- **Pattern**: Based on Coursera Deep Learning Week 3

## License

MIT License - Free to use, modify, and distribute.