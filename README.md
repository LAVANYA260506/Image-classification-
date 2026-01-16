# Image Classification using CNN (PyTorch)

This project implements an **Image Classification system using a Convolutional Neural Network (CNN)** built with **PyTorch**.  
The model is trained and evaluated using **Google Colab with GPU support** and is capable of making predictions on real-world images.

---

## 📌 Project Overview

The goal of this project is to understand and implement the complete **CNN pipeline**, including:
- Data preprocessing
- Model building
- Training and evaluation
- Debugging common deep learning issues
- Performing inference on new images

This project focuses on **concept clarity and hands-on debugging**, rather than just achieving high accuracy.

---

## 🧠 Model Details

- Custom CNN architecture using:
  - Convolutional layers
  - ReLU activation
  - Max pooling
  - Fully connected layers
- Loss Function: `CrossEntropyLoss`
- Optimizer: **Stochastic Gradient Descent (SGD)**
- Training performed on GPU using Google Colab

---

## 📊 Results

- Training loss decreased consistently over epochs
- Final test accuracy: **~68%**
- Successfully predicts classes for unseen real images (e.g., dog, frog)

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- Google Colab (GPU)
- PIL (for image loading)

---

## 🚀 How to Run the Project

1. Open the notebook `image_classification_CNN.ipynb` in **Google Colab**
2. Enable GPU:
   - Runtime → Change runtime type → GPU
3. Run all cells in order
4. (Optional) Upload your own images and test predictions

---

## 🔍 Key Learning Outcomes

- Understanding how CNNs process image data
- Importance of correct preprocessing and normalization
- How GPU acceleration improves training performance
- Debugging common issues such as:
  - Kernel crashes on local machines
  - CPU-only PyTorch installation
  - Incorrect accuracy calculation
- Performing real image inference using a trained CNN

---

## ⚠️ Challenges Faced & Fixes

- **Kernel crashing on laptop** → Switched to Google Colab GPU
- **CUDA not detected** → Fixed runtime type and PyTorch setup
- **Very low accuracy initially** → Fixed evaluation logic
- **Incorrect real-image predictions** → Ensured correct transforms and RGB conversion

These challenges helped build a strong understanding of real-world deep learning workflows.

---

## 📁 Repository Contents

- `image_classification_CNN.ipynb` – Main CNN training and inference notebook
- `README.md` – Project documentation

---

## ✨ Future Improvements

- Add data augmentation
- Improve CNN architecture
- Tune SGD learning rate and momentum
- Track validation loss and accuracy
- Deploy the trained model using a simple interface

---

## 🙌 Acknowledgement

This project was developed while learning **Convolutional Neural Networks and Deep Learning**, following tutorials and reinforcing concepts through independent debugging and experimentation.

---

## 📬 Author

**Lavanya**  
Student | Learning Deep Learning & Machine Learning
