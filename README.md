Indian Food Image Classifier
This project involves building a deep learning-based food image classifier that identifies 150 different classes of Indian food items. The classifier uses advanced hybrid architectures and optimization techniques to achieve high accuracy and robustness on a diverse dataset of Indian food images.

Table of Contents
Introduction
Dataset
Model Architecture
Overcoming Overfitting
Preprocessing
Training
Evaluation
Usage
Results
Future Work
Contributing
License
Introduction
Food classification is an essential task in applications like recipe recommendation, dietary analysis, and restaurant automation. This project aims to classify Indian food images into 150 categories using state-of-the-art deep learning architectures. Through extensive experimentation, we developed hybrid models and explored various loss functions, optimizers, and regularization techniques to achieve optimal performance.

Dataset
The dataset contains images of 150 Indian food categories, such as:

Biryani, Butter Chicken, Gulab Jamun, Dosa, Idli, etc.
Structure:
Training Set: ~70% of the images.
Validation Set: ~20% of the images.
Test Set: ~10% of the images.
The dataset underwent preprocessing to ensure uniformity in image size and quality.

Model Architecture
We adopted a hybrid approach combining the strengths of multiple architectures:

Vision Transformer (ViT):

Leveraged for its ability to capture global relationships in images.
Fine-tuned on the dataset.
EfficientNet:

Incorporated for its efficient scaling and feature extraction capabilities.
ConvNeXt:

Added to enhance local feature learning with modern CNN design principles.
Final Layers:

Combined extracted features from these models using concatenation and a series of dense layers.
Dropout layers were added for regularization, followed by a softmax activation for 150-class classification.
Overcoming Overfitting
To address overfitting, we implemented the following strategies:

Data Augmentation:

Applied random rotations, horizontal flips, zoom, and brightness adjustments during training.
Dropout Layers:

Added dropout with rates between 0.3 and 0.5 in the dense layers.
Early Stopping:

Monitored validation loss and stopped training once no improvement was observed.
Weight Regularization:

Included L2 regularization in dense layers to penalize overly complex weights.
Model Fine-Tuning:

Gradually unfroze pre-trained layers while using a low learning rate to avoid overfitting.
Preprocessing
Image Resizing:
Resized all images to 224x224 pixels.
Normalization:
Scaled pixel values to the range [0, 1].
Class Balancing:
Oversampled minority classes to ensure balanced training.
Training
We explored various loss functions and optimizers to achieve optimal results:

Loss Functions:

Sparse Categorical Crossentropy: Worked best for the dataset.
Focal Loss: Tested to address class imbalance, but simpler methods sufficed.
Optimizers:

Adam: Primary optimizer with dynamic learning rate scheduling.
SGD with Momentum: Tested for stability and improved convergence.
Hyperparameter Tuning:

Tuned batch size, learning rate, and dropout rates using grid search.
Batch Size: 32.

Epochs: 50 (with early stopping).

Evaluation
The model was evaluated on the test set using the following metrics:

Accuracy: Overall classification performance.
Precision, Recall, F1-Score: Detailed per-class analysis.
Confusion Matrix: To visualize misclassifications.
Usage
Requirements
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Inference
To classify an image:

Place the image in the test_images/ directory.
Run the inference script:
bash
Copy code
python predict.py --image test_images/sample.jpg
Output:
Predicted class label.
Confidence score.
Results
Accuracy: ~92% on the test set.
Inference Time: ~100ms per image on a GPU.
Future Work
Extend the classifier to include global cuisines.
Deploy the model as a web application for real-world use.
Experiment with advanced architectures like Swin Transformers and ensemble techniques.
Contributing
Contributions are welcome! Please submit a pull request or report issues in the repository.

License
This project is licensed under the MIT License. See the LICENSE file for details.
