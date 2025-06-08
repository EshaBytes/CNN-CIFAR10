# 🧠 CNN Implementation on CIFAR-10 Dataset

This project presents a Convolutional Neural Network (CNN) developed using TensorFlow and Keras for image classification on the CIFAR-10 dataset. The model is trained with data augmentation to enhance generalization and evaluated using key performance metrics.

## 📌 Overview

- **Goal:** Classify 60,000 color images into 10 distinct classes (e.g., airplane, car, bird, etc.)
- **Approach:** A custom CNN architecture with three convolutional layers, max-pooling, and fully connected layers.
- **Tools:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## 🧾 Features

- Data preprocessing and normalization
- Data augmentation using `ImageDataGenerator`
- Multi-layer CNN architecture with ReLU and softmax activations
- Training with Adam optimizer and sparse categorical crossentropy loss
- Evaluation with accuracy, confusion matrix, and classification report
- Visualization of training history (loss/accuracy plots)

## 📊 Dataset

- **Source:** Built-in `cifar10` dataset from `tensorflow.keras.datasets`
- **Size:** 60,000 32x32 color images in 10 classes (50,000 training + 10,000 test)

## 📁 Files

| File Name                       | Description                                 |
|--------------------------------|---------------------------------------------|
| `cnn_cifar10_implementation.ipynb` | Main Jupyter notebook with model code      |
| `README.md`                    | Project overview and instructions           |

## ⚙️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/CNN-CIFAR10.git
   cd CNN-CIFAR10
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
3. Run the notebook:
   - Use Jupyter Notebook or Google Colab to open cnn_cifar10_implementation.ipynb
   - Execute all cells
  
## 📈 Results
- Achieved high accuracy on both training and test sets.
- Evaluated with confusion matrix and detailed classification report.
- Plotted learning curves for better understanding of model performance.

## 🧠 Sample Output
![Screenshot (289)](https://github.com/user-attachments/assets/b0b501ff-974d-47a0-8fad-b19dcec357ed)
![Screenshot (290)](https://github.com/user-attachments/assets/163b6eb7-9b1e-4752-9e1f-75e97123dab6)
![Screenshot (291)](https://github.com/user-attachments/assets/b9fb9c66-f8b0-4bcd-bc1d-2df9ed20c673)
![Screenshot (292)](https://github.com/user-attachments/assets/c30ca407-0822-4e78-a748-45ba328c14f1)

