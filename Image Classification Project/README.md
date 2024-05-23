# Deep Learning Image Classification Project

This project aims to classify images of dermatological conditions using the FITZPATRICK17k dataset. The dataset contains images of multiple dermatological conditions, and the goal is to build and evaluate several convolutional neural networks (CNN) for accurate classification. The project is implemented using TensorFlow and Keras libraries, and consists of the following main phases:

1. Exploratory Data Analysis (EDA) and Preprocessing
2. Baseline Model Development
3. Transfer Learning with VGG-16 and VGG-19
4. Transfer Learning with InceptionV3
5. Transfer Learning with ResNet50V2
6. Metrics Evaluation

## Files

The project is organized into several Jupyter Notebooks, each focusing on different aspects of the workflow:

- **1 - EDA and Preprocessing.ipynb**: This notebook covers the initial exploration and preprocessing of the dataset, including data extraction using URLs, stratified data splitting, and feature engineering.
- **2 - Baseline Model.ipynb**: This notebook builds a baseline CNN model from scratch using Keras to provide a reference point for model performance.
- **3 - VGG-19 and VGG-16.ipynb**: This notebook applies transfer learning techniques using the VGG-16 and VGG-19 architectures to improve classification performance.
- **4 - InceptionV3.ipynb**: This notebook implements the InceptionV3 architecture using transfer learning to enhance model accuracy.
- **5 - ResNet50V2.ipynb**: This notebook uses the ResNet50V2 architecture with transfer learning and further fine-tunes the model using Keras Tuner for hyperparameter optimization.
- **6 - Metrics Evaluation.ipynb**: This notebook evaluates the performance of all models based on various metrics, with a focus on the weighted F1-score.

Additionally, the dataset file `fitzpatrick17k.csv` is included, containing the image URLs and metadata.

## Dataset

The FITZPATRICK17k dataset contains images of various dermatological conditions. The images are used to train and test the CNN models for classification tasks. The dataset is split into training, validation and test sets in a stratified manner to maintain the class distribution.


## Future Work

The project is a work in progress and still requires some refinements. For example, in the current implementation, layer freezing is applied during transfer learning to prevent weights from changing during training. However, typically in transfer learning, the top few layers should remain unfrozen to fine-tune the top layers on the new dataset. This aspect will be refined in future updates.

## How to Run

1. Clone the repository.
2. Ensure you have the required dependencies installed (`tensorflow`, `keras`, `pandas`, etc.).
3. Run the notebooks in the order specified to reproduce the results.

**Note**: when replicating this code, it's possible to have some versioning issues with TensorFlow imports. The reason behind is that I want to use the GPU/CUDA and TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Ensure compatibility of Pyhton, TensorFlow and Keras versions with the code provided in the notebooks. For more information check: https://www.tensorflow.org/install/pip#windows-native


- LinkedIn - [Diogo Pires](https://www.linkedin.com/in/diogo-f-m-pires)
- GitHub: [diogo-pires-github](https://github.com/diogo-pires-github)