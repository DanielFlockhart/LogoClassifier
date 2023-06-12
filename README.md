# Logo Classifier

## Overview

The Image classifier is a Python-based machine learning project that utilizes TensorFlow to predict a company logo from an input image. For the purposes of this repo, my dataset and validation data are the same and only have 1 image per class. Naturally this is by no means a good idea in practice, but for demonstration purposes it works.

The core of the project is a deep learning model built using the TensorFlow framework. The model is trained on a dataset comprising the logos of the top 20 most valued companies, enabling it to accurately identify and classify the logos. It uses the ResNet50 CNN architecture and uses said pre-trained weights. By leveraging convolutional neural networks (CNNs), the project achieves reliable logo recognition and can handle various logo variations and distortions.

## Features

- **Logo Prediction**: Given an input image, the model predicts the company logo from the top 20 most valued companies.
- **Simple Classification**: The project focuses on recognizing logos from a specific set of companies, allowing for straightforward classification.
- **High Accuracy**: The deep learning model achieves high prediction accuracy by leveraging state-of-the-art techniques and pre-trained models.


## Installation

To set up the Logo Classifier project locally, follow these steps:

1. Clone the GitHub repository:

   ```
   $ git clone https://github.com/DanielFlockhart/logoclassifier.git
   ```

2. Navigate to the project directory:

   ```
   $ cd logoclassifier
   ```

3. Install the project dependencies:

   ```
   $ pip install -r requirements.txt
   ```

4. Download the pre-trained model weights (if available) and place them in the appropriate directory (e.g., `models/`), or train a new model with your data.
   ```
   $ cd src/training
   $ python train.py
   ```


5. Start the application:

   ```
   $ python main.py
   ```

6. Use the project by providing an input image and observing the predicted company logo.

## Usage

The Logo Classifier allows you to predict the company logo from an input image. Follow these steps to use the project:

1. Prepare an image containing a company logo.

2. Run the `main.py` script:

   ```
   $ python main.py
   ```

3. Provide the path or filename of the input image when prompted.

4. The model will process the image and predict the ogo.

5. The predicted logo will be displayed on the screen.

## Contributing

Contributions to the Logo Classifier project are highly appreciated.