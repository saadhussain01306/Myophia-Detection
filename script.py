import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the trained model
model_path = "myopia_classifier_model.h5"
model = load_model(model_path)

# Define class names
class_labels = ['Myopia', 'Normal']

# Parameters
img_height, img_width = 224, 224

def predict_image(image):
    """
    Preprocesses the uploaded image and predicts the class using the trained model.
    Args:
        image: Uploaded image.
    Returns:
        str: Predicted class label.
        float: Confidence score of the prediction.
    """
    resized_image = cv2.resize(image, (img_height, img_width))
    preprocessed_image = preprocess_input(resized_image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index] * 100
    predicted_class_label = class_labels[predicted_class_index]


    return predicted_class_label, confidence_score

# Function to upload and process image
def upload_and_predict():
    """
    Opens a file dialog to upload an image, predicts its class, and displays the result in the terminal.
    """
    # Open a file dialog to select an image
    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        # Load and process the image
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Unable to read the selected image.")
            return
        
        predicted_label, confidence = predict_image(image)
        print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}%")
    else:
        print("No image was selected.")

# Run the upload and predict function
if __name__ == "__main__":
    upload_and_predict()
