# Quantum_Computing/extract_features.py

import os
from PIL import Image
import numpy as np

# Assuming IMAGE_WIDTH, IMAGE_HEIGHT might come from config or be passed
# For a standalone script, it's safer to define them or import them if intended.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def extract_features_from_image(image_path, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT), convert_to="L"):
    """
    Extracts pixel features from an image.
    :param image_path: Path to the image file.
    :param img_size: Tuple (width, height) to resize the image to.
    :param convert_to: Mode to convert the image to ('L' for grayscale, 'RGB' for color).
    :return: Flattened numpy array of pixel values, or None if image cannot be processed.
    """
    try:
        with Image.open(image_path) as img:
            img = img.resize(img_size)
            if convert_to:
                img = img.convert(convert_to)
            
            # Convert image data to numpy array and flatten
            features = np.array(img).flatten()
            return features
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    # Assuming you have a dummy 'data' directory for testing
    # You would typically call this from another script.
    
    # Create a dummy image for testing
    dummy_img_path = "dummy_test_image.png"
    if not os.path.exists(dummy_img_path):
        dummy_img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color = 128) # Grey image
        dummy_img.save(dummy_img_path)
        print(f"Created dummy image: {dummy_img_path}")

    features = extract_features_from_image(dummy_img_path)
    if features is not None:
        print(f"Extracted features shape: {features.shape}")
        print(f"First 10 features: {features[:10]}")
    else:
        print("Failed to extract features from dummy image.")

    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path) # Clean up
        print(f"Removed dummy image: {dummy_img_path}")