# Object Detection with OpenCV + ImageAI

This project captures images from the **webcam** using **OpenCV**, saves them locally, and performs **object detection** with a pre-trained RetinaNet model (`resnet50_coco_best_v2.0.1.h5`) via the **ImageAI** library.  
It provides a simple way to test real-time computer vision pipelines.

---

## ✨ Features
- 📷 Capture frames from webcam with OpenCV  
- 🖼️ Save snapshots with the `SPACE` key  
- 🧠 Run object detection using ImageAI + RetinaNet (ResNet50 backbone)  
- 📊 Display detection results (object name + probability) in the terminal  
- 💾 Save annotated images with bounding boxes

---

## 🧱 Requirements
- Python 3.7–3.9 (ImageAI compatibility)
- [OpenCV](https://opencv.org/)  
- [ImageAI](https://github.com/OlafenwaMoses/ImageAI)  
- [TensorFlow 2.x](https://www.tensorflow.org/)  
- Pre-trained RetinaNet model:  
  Download [`resnet50_coco_best_v2.0.1.h5`](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5)  
  Place it in the project root directory.

Install dependencies:
```bash
pip install opencv-python imageai tensorflow
```

---

## ▶️ Usage
Run the script:
```bash
python detect.py
```

Controls:
- `ESC` → exit  
- `SPACE` → capture image and run object detection  

---

## 📂 Project Structure
```
.
├── detect.py
├── resnet50_coco_best_v2.0.1.h5
└── images/
    ├── object.jpg       # raw capture
    └── newObject.jpg    # annotated output
```

---

## ✅ Example Output
Terminal:
```
person  :  99.32
bicycle :  87.45
dog     :  76.12
```

Image (`images/newObject.jpg`)
