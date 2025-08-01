import os

# Root directory for data (assuming it's relative to the PROJECT root)
# Adjust this path based on where 'data' folder is relative to where your script is run
# Let's assume DATA_ROOT_DIR is at the same level as Quantum_Computing
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This is Quantum_Computing/

DATA_ROOT_DIR = os.path.join(BASE_DIR, 'data') # <-- UPDATED LINE: Points to Quantum_Computing/data

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# List of tumor types (must include 'notumor' for binary classification)
TUMOR_TYPES_LIST = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Dummy data generation for development/testing
# Set to 0 to use real data, > 0 to generate N dummy images per class
DUMMY_IMAGE_COUNT = 0 # Set to 200 for dummy data testing (if you want to test without real images)

# Paths for saving classical model components
CLASSICAL_MODELS_DIR = os.path.join(BASE_DIR, 'classical_model_results')
os.makedirs(CLASSICAL_MODELS_DIR, exist_ok=True) # Ensure directory exists

PCA_SAVE_PATH = os.path.join(CLASSICAL_MODELS_DIR, 'pca_model.pkl')
SCALER_SAVE_PATH = os.path.join(CLASSICAL_MODELS_DIR, 'minmax_scaler.pkl')
LABEL_ENCODER_SAVE_PATH = os.path.join(CLASSICAL_MODELS_DIR, 'label_encoder.pkl') # For classical multi-class

# Paths for saving quantum model components
QML_RESULTS_DIR = os.path.join(BASE_DIR, 'model_results')
os.makedirs(QML_RESULTS_DIR, exist_ok=True) # Ensure directory exists

QML_PARAMS_SAVE_PATH = os.path.join(QML_RESULTS_DIR, 'qml_trained_params.pkl')
QML_QUANTUM_SCALER_SAVE_PATH = os.path.join(QML_RESULTS_DIR, 'qml_quantum_scaler.pkl')