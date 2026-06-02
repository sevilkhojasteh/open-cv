Week 1: Computer Vision Foundations
├─ Learn YOLOv8 architecture basics
│ Resources: Ultralytics docs, "Zero to Mastery CV" (YouTube)
├─ Train a custom model on Roboflow dataset
│ Dataset: Helmet detection or PPE detection (industrial context)
└─ Test locally on your laptop
Goal: 90%+ accuracy on validation set

Week 2: Model Optimization for Edge
├─ Learn quantization (FP32 → INT8)
│ Resources: TensorFlow Lite documentation
├─ Convert model to TFLite or ONNX
├─ Benchmark model size reduction & speed improvement
└─ Export optimized model

Week 3: Embedded Deployment
├─ Set up Raspberry Pi 4 / Jetson Nano
│ (Or use ESP32-CAM if budget constrained)
├─ Install OpenCV, TFLite runtime
├─ Write Python inference script
│ - Capture camera feed
│ - Run detection (aim for >10 FPS)
│ - Draw bounding boxes
└─ Optimize performance (multi-threading, resolution tuning)

Week 4: IoT Integration + Dashboard
├─ Set up MQTT broker (Mosquitto)
├─ Publish detection events (JSON format)
├─ Build web dashboard (React + Chart.js)
│ - Real-time detection feed
│ - Statistics (objects detected per hour)
│ - Alert system
└─ Deploy dashboard (Vercel/Netlify)
