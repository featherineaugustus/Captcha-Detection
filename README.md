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

## Pipeline
1. Load image
2. Crop image on the top, bottom, left and right to only expose the 5 characters
3. Remove all pixels that are not black (background), and invert the color back to BW
4. Slice the image into 5 equal parts (9 pixels width each)
5. Perform character predicition for each of the 5 image parts
6. Combine the predicted characterse in sequence back into a string
7. Return string as output

## Usage
1. Training (see `main.ipynb`)
    - Prepare your dataset
    - Store captcha images in data/.
    - Create a CSV data.csv with columns path, value for each character image.
    - Run the training script to train the CNN model.
        - In this case, we are using a CNN_Model with 2 convolutional layer and 2 dense layer.
        - The images are resized to 28 X 28 and normalized before training.
        - The loss function is the Cross Entropy, and optimized with Adam
        - GPU may be enabled during training.

    - For this work, we take all the provided data to train the model, as some characters only have a single count for training purposes.
        - However, if we were to use a smaller epoch, some characters may be predicted wrongly.

2. Inference (see `main.ipynb`)
    - Use the provided `Captcha` class at the endto predict the captcha text from an image:

   ```python
    from captcha import Captcha

    captcha = Captcha(model_path='captcha_model.pth')
    predicted_text = captcha('images/input01.jpg', 'pred/output01.txt')
    print("Predicted captcha:", predicted_text)
    ```

## Code Explanation
- `Captcha` class: Handles loading the model, image preprocessing, slicing the image into characters, and predicting each character.
- `CNN_Model`: The CNN architecture used for character classification.
- `crop_image` and `slice_left_to_right`: Utility functions for image cropping and segmentation.
- `predict_character`: Method to predict single characters from image slices.

## Evaluation
- Accuracy can be computed by comparing predicted captcha texts against ground truth labels.
    - In this case, we obtained an accuracy of 100%, but then it may be because it is overfitted.
- Use the provided CSV logs and utility scripts to analyze incorrect predictions.
- While we know that the model is overfitted to the training set now, but as the formatting of the characters is always the same, it is alright.

## Future Work
- Extend to handle variable-length captchas, symbols, and other language characters.
- Improve segmentation with more robust image processing.
- Use sequence models (e.g., RNNs, Transformers) for end-to-end captcha recognition.
- Incorporate data augmentation to improve model robustness.