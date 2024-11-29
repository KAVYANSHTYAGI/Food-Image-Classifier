# **Indian Food Image Classifier**

This project involves building a deep learning-based food image classifier capable of identifying 150 different classes of Indian food items. The project explores various state-of-the-art architectures, including hybrid models, to achieve high accuracy and robust performance on a complex dataset.

---

## **Introduction**

Food classification is an essential task in applications like recipe recommendation, dietary analysis, and restaurant automation. This project classifies Indian food images into 150 categories using advanced architectures. The model integrates features from multiple Convolutional Neural Networks (CNNs) and Transformer-based models for improved accuracy and robustness.

---

## **Dataset**

The dataset consists of 150 classes of Indian food items, including popular dishes like Biryani, Butter Chicken, Gulab Jamun, Dosa, and Idli.

### Dataset Structure:
- **Training Set**: ~70% of the images.
- **Validation Set**: ~20% of the images.
- **Test Set**: ~10% of the images.

Each image is preprocessed to maintain consistency in quality and size.

---

## **Model Architecture**

To achieve state-of-the-art performance, we utilized the following architectures:

### **Core Architectures**
1. **Vision Transformer (ViT)**: For capturing global relationships in images using self-attention.
2. **EfficientNet (B7)**: For efficient scaling and feature extraction.
3. **ConvNeXt**: A modern CNN architecture combining local feature extraction with strong generalization.
4. **ResNet (50, 101)**: For capturing residual connections to prevent vanishing gradients.
5. **MobileNetV3**: Lightweight and optimized for mobile and low-resource applications.
6. **DenseNet (201)**: For efficient feature reuse through dense connections.
7. **InceptionV3 & InceptionResNetV2**: For multi-scale feature extraction and residual learning.
8. **Xception**: For efficient learning using depthwise separable convolutions.
9. **NASNet-Large**: Designed using Neural Architecture Search for high accuracy.
10. **VGG (16, 19)**: Classical architectures used as baselines for comparison.
11. **Swin Transformer**: Hierarchical Transformer with local and global feature learning.

### **Hybrid Architectures**
We experimented with various hybrid combinations to leverage the strengths of multiple models:
- **EfficientNet + ResNet-50 + ViT**: Combined global features from ViT with local features from EfficientNet and ResNet.
- **ConvNeXt + InceptionV3 + DenseNet-201**: Merged ConvNeXt's local feature extraction with DenseNet's gradient flow and InceptionV3's multi-scale capabilities.
- **MobileNetV3 + ViT**: Balanced lightweight computation with global feature representation.

### **Feature Fusion**
- Extracted features from each architecture were concatenated.
- Fully connected layers refined the combined features.
- Dropout and L2 regularization prevented overfitting.
- The final dense layer with softmax activation outputted probabilities for 150 classes.

---

## **Overcoming Overfitting**

To ensure the model generalized well to unseen data:
1. **Data Augmentation**: Techniques like random rotation, flipping, zoom, and brightness adjustment.
2. **Dropout Layers**: Added dropout with rates between 0.3 and 0.5.
3. **Early Stopping**: Stopped training based on validation loss to avoid overfitting.
4. **L2 Regularization**: Penalized large weights to prevent overfitting.
5. **Gradual Unfreezing**: Fine-tuned pre-trained layers gradually with a low learning rate.

---

## **Preprocessing**

1. **Image Resizing**: Resized all images to 224x224 pixels.
2. **Normalization**: Scaled pixel values to the range [0, 1].
3. **Class Balancing**: Oversampled minority classes to ensure balanced training.

---

## **Training**

We tested various loss functions and optimizers:
1. **Loss Functions**:
   - Sparse Categorical Crossentropy: Best performance on this dataset.
   - Focal Loss: Tested to address class imbalance but not necessary for this dataset.
2. **Optimizers**:
   - Adam: Primary optimizer with dynamic learning rate scheduling.
   - SGD with Momentum: Tested for stability and improved convergence.

### Training Parameters:
- **Batch Size**: 32.
- **Epochs**: 50 (with early stopping).
- **Learning Rate Scheduler**: Adjusted dynamically during training.

---

## **Evaluation**

The model was evaluated using the following metrics:
- **Accuracy**: Overall classification performance.
- **Precision, Recall, F1-Score**: Detailed per-class analysis.
- **Confusion Matrix**: Visualized misclassifications.

---

## **Usage**

### **Requirements**
Install the required dependencies:

```bash
pip install -r requirements.txt
