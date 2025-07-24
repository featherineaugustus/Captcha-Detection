# Captcha Recognition Project

## Overview
This project implements a simple AI-based captcha recognition system that detects and predicts characters in 5-character captchas composed of uppercase letters (A-Z) and digits (0-9). The solution uses a Convolutional Neural Network (CNN) trained on a custom dataset and performs inference by segmenting the captcha image into individual characters and classifying each character separately.

## Features
- Handles fixed-length (5-character) captchas with consistent font, spacing, and color scheme.
- Custom CNN model for character classification trained on user-provided labeled images.
- Image preprocessing including cropping, thresholding, and segmentation into individual character slices.
- Inference pipeline wrapped into a reusable `Captcha` class with a simple interface.
- Outputs predicted captcha text to a file for easy integration.

## Project Structure
```bash
├── sampleCaptchas
│   ├── data            # Data for training CNN
│   ├── input           # Input image folder
│   ├── output          # Output txt folder
│   ├── pred            # Predicted results txt folder
│   ├── data_pred.csv   # Predicted results
│   └── data.csv        # Data for training CNN summarized
│
├── captcha_model.pth # Trained CNN model weights
├── .gitignore
├── main.ipynb
└── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/featherineaugustus/Captcha-Detection
   cd Captcha-Detection
   ```


## Usage
1. Training
    - Prepare your dataset
    - Store captcha images in data/.
    - Create a CSV data.csv with columns path, value for each character image.
    - Run the training script to train the CNN model.

2. Inference
    - Use the provided Captcha class to predict the captcha text from an image:

   ```python
    from captcha import Captcha

    captcha = Captcha(model_path='captcha_model.pth')
    predicted_text = captcha('images/input01.jpg', 'pred/output01.txt')
    print("Predicted captcha:", predicted_text)
    ```

## Code Explanation
- Captcha class: Handles loading the model, image preprocessing, slicing the image into characters, and predicting each character.

- SimpleCNN: The CNN architecture used for character classification.

- crop_image and slice_left_to_right: Utility functions for image cropping and segmentation.

- predict_character: Method to predict single characters from image slices.

## Evaluation
- Accuracy can be computed by comparing predicted captcha texts against ground truth labels.
- Use the provided CSV logs and utility scripts to analyze incorrect predictions.
- While we know that the model is overfitted to the training set now, but as the formatting of the characters is always the same, it is alright.

## Future Work
- Extend to handle variable-length captchas.
- Improve segmentation with more robust image processing.
- Use sequence models (e.g., RNNs, Transformers) for end-to-end captcha recognition.
- Incorporate data augmentation to improve model robustness.