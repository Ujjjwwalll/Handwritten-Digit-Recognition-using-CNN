# 🔢 Handwritten Digit Recognition using CNN

A Deep Learning-based Handwritten Digit Recognition System built using **Convolutional Neural Networks (CNN), TensorFlow, OpenCV, and Streamlit**. The application allows users to upload images of handwritten digits (0–9) and receive real-time predictions with confidence scores through an interactive web interface.

## 📌 Project Overview

Handwritten digit recognition is a computer vision problem in which a machine learning model identifies numerical digits from images.

In this project, I developed and trained a CNN model using the **MNIST handwritten digits dataset** and integrated it with a Streamlit web application to recognize handwritten digits from externally uploaded images.

The application performs image preprocessing using OpenCV before passing the processed image to the trained CNN model for classification.

## 🎯 Project Objectives

- Understand and implement CNN architecture for image classification.
- Work with image datasets and perform image preprocessing.
- Extract meaningful visual features using convolutional and pooling layers.
- Train, evaluate, and save a CNN model using TensorFlow/Keras.
- Integrate a trained deep learning model into a Streamlit application.
- Perform predictions on user-uploaded handwritten digit images.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow & Keras | CNN model development and training |
| OpenCV | Image preprocessing and manipulation |
| NumPy | Numerical computations and array operations |
| Matplotlib | Image visualization and performance analysis |
| Streamlit | Interactive web application |
| MNIST | Handwritten digit dataset |
| Git & GitHub | Version control and project management |

## 📊 Dataset: MNIST

The project uses the MNIST dataset, a widely used benchmark dataset for handwritten digit recognition.

**Dataset details:**
- Total images: 70,000
- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Image format: Grayscale
- Pixel intensity: 0–255
- Classes: 10 (digits 0–9)

The dataset is loaded directly using TensorFlow/Keras.

## 🧠 CNN Model Architecture

A Sequential CNN architecture was implemented using TensorFlow/Keras.

**Model architecture:**

1. **Convolutional Layer 1**
   - 32 filters
   - Kernel size: 3 × 3
   - ReLU activation

2. **Max Pooling Layer 1**
   - Pool size: 2 × 2

3. **Convolutional Layer 2**
   - 64 filters
   - Kernel size: 3 × 3
   - ReLU activation

4. **Max Pooling Layer 2**
   - Pool size: 2 × 2

5. **Flatten Layer**
   - Converts feature maps into a one-dimensional vector.

6. **Dense Layer**
   - 64 neurons
   - ReLU activation

7. **Output Layer**
   - 10 neurons
   - Softmax activation for digit classification.

**Training configuration:**

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 10
- Batch Size: 64
- Validation Split: 10%

## ⚙️ Image Preprocessing Pipeline

One of the important aspects of this project is handling external handwritten images.

Unlike MNIST images, uploaded photographs may contain different backgrounds, lighting conditions, ink colors, and stroke thicknesses.

To handle these differences, an image preprocessing pipeline was implemented using OpenCV.

**Preprocessing workflow:**

1. **Image Upload:** Accepts JPG, JPEG, and PNG images through Streamlit.
2. **Color Processing:** Converts the uploaded image into RGB and HSV representations.
3. **Grayscale / Thresholding:** Separates the handwritten digit from its background using thresholding.
4. **Noise Reduction:** Applies Gaussian Blur to reduce unwanted noise.
5. **Morphological Operations:** Uses morphological closing and dilation to improve stroke continuity and thickness.
6. **Contour Detection:** Identifies the largest contour and extracts the digit.
7. **Image Resizing:** Resizes the extracted digit while maintaining its aspect ratio.
8. **Centering:** Places the digit on a 28 × 28 black canvas.
9. **Normalization:** Scales pixel values from 0–255 to 0–1.
10. **Reshaping:** Converts the image into the required CNN input shape of (1, 28, 28, 1).

The processed image is then passed to the trained CNN model for prediction.

## 🚀 Application Features

- Interactive Streamlit web interface.
- Upload handwritten digit images directly from a local device.
- Automatic image preprocessing using OpenCV.
- Real-time digit classification (0–9).
- Prediction confidence score.
- Probability distribution visualization for all 10 classes.
- Display of the final processed 28 × 28 image.
- Reuses the saved CNN model without retraining for every prediction.

## 📈 Model Performance

**MNIST Test Accuracy: Approximately 99.81%**

The model achieved approximately 99.81% accuracy on the MNIST test dataset.

The application was also tested with external handwritten images to examine its ability to generalize beyond the original dataset.

*Note: MNIST test accuracy and prediction confidence on externally uploaded photographs are different evaluation measures. External-image performance can vary depending on handwriting style and image quality.*

## 📂 Project Structure

```text
CNN-Digit-Recognition/
│
├── app.py
│
├── mnist_cnn_model.keras
│
├── requirements.txt
│
├── README.md
```

## 🔮 Future Improvements

- Improve generalization across different handwriting styles.
- Train with image augmentation to handle variations in handwriting.
- Improve image preprocessing for different lighting and background conditions.
- Extend the application to recognize multiple digits in a single image.
- Explore transfer learning and more advanced CNN architectures.
- Deploy the application for public access.

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Understanding and implementing CNN architecture.
- Working with image datasets and image classification.
- Convolution, pooling, flattening, and fully connected layers.
- Image preprocessing and morphological operations using OpenCV.
- Model training, validation, and evaluation.
- Saving and loading trained deep learning models.
- Integrating a CNN model with a Streamlit web application.
- Performing inference on external user-uploaded images.

## 👨‍💻 Author

**Ujjwal Tripathi**

BCA Graduate | Aspiring AI/ML Developer

Interested in Machine Learning, Deep Learning and Generative AI.

---

