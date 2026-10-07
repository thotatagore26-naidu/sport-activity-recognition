# Sport Activity Recognition using YOLOv11n on Sony IMX500

## Project Information

**Course:** Artificial Intelligence & Software Development

**Institution:** Deggendorf Institute of Technology (THD)

**Project Title:**
Sport Activity Recognition using Sony IMX500 AI Camera

Team Members

- Tagore Thotakura
- Charuphala Balasubramanian
- Varsha Palampalli

---

# Project Overview

This project implements a real-time sport activity recognition system using a Sony IMX500 AI camera connected to a Raspberry Pi 5.

A lightweight YOLOv11n object detection model was trained on a custom sports dataset and deployed directly onto the IMX500 edge AI sensor. Running inference on the camera eliminates the need for cloud processing, resulting in low latency, improved privacy, and efficient real-time performance.

---

# Objectives

- Build a custom object detection model for sports activities.
- Collect and prepare a custom dataset.
- Train a YOLOv11n model.
- Export the trained model to Sony IMX500 format.
- Convert the exported model into RPK format.
- Deploy the model on Raspberry Pi 5.
- Perform real-time on-device inference.

---

# Dataset

Dataset Name

Sports AISD Dataset

Dataset Size

916 annotated images

Classes

1. Basketball Shoot
2. Cricket Batting
3. Tennis Swing
4. Volleyball Spike

Dataset Split

- Training: 80%
- Validation: 10%
- Test: 10%

Image Resolution

640 × 640 pixels

Dataset Sources

- UCF101 sports action videos
- Self-recorded cricket batting video

The dataset was manually annotated using LabelImg in YOLO format.

---

# Model

Model

YOLOv11n

Training Platform

Google Colab (NVIDIA T4 GPU)

Training Epochs

79

Model Size

5.2 MB

Parameters

2.58 Million

Framework

Ultralytics YOLO

---

# Performance

mAP@50

95.7%

Precision

96.6%

Recall

90.2%

F1 Score

93.3%

---

# Deployment Pipeline

Dataset

↓

Frame Extraction

↓

Manual Annotation (LabelImg)

↓

YOLO Dataset Preparation

↓

YOLOv11n Training

↓

best.pt

↓

Export to Sony IMX Format

↓

packerOut.zip

↓

Convert to network.rpk

↓

Deploy on Raspberry Pi 5

↓

Real-Time Detection using Sony IMX500

---

# Folder Structure

Project_Submission/

Model/

- best.pt
- network.rpk
- labels.txt
- packerOut.zip
- model_imx.onnx
- model_imx.pbtxt
- dnnParams.xml

Source_Code/

- imx500_object_detection_demo.py
- README.md
- requirements.txt

Presentation/

- Sport_Activity_Recognition_final.pptx

---

# Hardware

- Raspberry Pi 5
- Sony IMX500 AI Camera

---

# Software

- Python
- Ultralytics YOLO
- OpenCV
- Picamera2
- Sony IMX500 Tools

---

# Running the Project

Install the required packages

```
pip install -r requirements.txt
```

Run the detection script

```
python imx500_object_detection_demo.py \
--model network.rpk \
--labels labels.txt
```

---

# Limitations

- Small dataset compared to production-scale systems.
- INT8 calibration uses limited calibration images.
- Accuracy decreases in crowded scenes with multiple players.
- Performance depends on camera angle and lighting conditions.

---

# Future Improvements

- Increase dataset size.
- Add additional sports.
- Integrate pose estimation.
- Improve INT8 calibration.
- Support multi-person tracking.

---

# Acknowledgement

This project was developed for the Artificial Intelligence & Software Development course at Deggendorf Institute of Technology.