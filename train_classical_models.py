# Quantum_Computing/train_classical_models.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime
from PIL import Image # For image processing

# --- Robust Imports from local package and config ---
try:
    # Use the local path for config, assuming train_classical_models.py is directly in Quantum_Computing/
    from config import (
        DATA_ROOT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, TUMOR_TYPES_LIST,
        PCA_SAVE_PATH, SCALER_SAVE_PATH, LABEL_ENCODER_SAVE_PATH, DUMMY_IMAGE_COUNT,
        CLASSICAL_MODELS_DIR
    )
    # The _extract_features_from_image function is usually a shared utility
    # It might be in quantum_model/preprocessing.py as used by QML training
    # For classical training, we'll put a local version if not easily importable or explicitly define
    from quantum_model.preprocessing import _extract_features_from_image # Assuming this is shared
    from quantum_model.evaluation import evaluate_model # Shared evaluation helper
except ImportError:
    # Fallback for imports if running standalone or path issues
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(current_dir, 'quantum_model'))
    try:
        from config import (
            DATA_ROOT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, TUMOR_TYPES_LIST,
            PCA_SAVE_PATH, SCALER_SAVE_PATH, LABEL_ENCODER_SAVE_PATH, DUMMY_IMAGE_COUNT,
            CLASSICAL_MODELS_DIR
        )
        from preprocessing import _extract_features_from_image # Local import within quantum_model sub-package
        from evaluation import evaluate_model # Local import within quantum_model sub-package
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import config, preprocessing, or evaluation. Error: {e}")
        sys.exit(1)


def load_and_preprocess_classical_data():
    """
    Loads image data, extracts features, applies PCA, and scales for classical ML models.
    Also saves PCA model and scaler for later use by QML.
    """
    print("\n--- Loading and Preprocessing Data for Classical ML ---")

    all_features = []
    all_labels = [] # Will store string labels

    if DUMMY_IMAGE_COUNT > 0:
        print(f"\n--- DUMMY DATA MODE ACTIVE (DUMMY_IMAGE_COUNT = {DUMMY_IMAGE_COUNT}) ---")
        print("Generating synthetic features for classical ML. Model performance will not be meaningful.")
        feature_vector_size = IMAGE_WIDTH * IMAGE_HEIGHT
        num_samples_per_class = DUMMY_IMAGE_COUNT // len(TUMOR_TYPES_LIST)
        
        for tumor_type_str in TUMOR_TYPES_LIST:
            for _ in range(num_samples_per_class):
                features = np.random.rand(feature_vector_size) # Random features
                all_features.append(features)
                all_labels.append(tumor_type_str)
        print(f"--- Generated {len(all_features)} dummy samples. ---\n")

    else: # Process actual data
        print(f"\n--- REAL DATA MODE ACTIVE (DUMMY_IMAGE_COUNT = 0) ---")
        print(f"Attempting to load real images from: {DATA_ROOT_DIR}")

        # Load from both Training and Testing directories
        for dataset_type in ['Training', 'Testing']:
            dataset_path = os.path.join(DATA_ROOT_DIR, dataset_type)
            if not os.path.exists(dataset_path):
                print(f"Warning: Directory not found: {dataset_path}. Skipping {dataset_type} data.")
                continue

            for tumor_type_str in TUMOR_TYPES_LIST:
                class_path = os.path.join(dataset_path, tumor_type_str)
                if not os.path.exists(class_path):
                    print(f"Warning: Class directory not found: {class_path}. Skipping.")
                    continue

                print(f"Loading images from: {class_path} for class: {tumor_type_str}")

                image_files = [
                    img_name for img_name in os.listdir(class_path)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ]
                if image_files:
                    for img_name in image_files:
                        img_path = os.path.join(class_path, img_name)
                        features = _extract_features_from_image(img_path, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT), convert_to="L")
                        if features is not None:
                            all_features.append(features)
                            all_labels.append(tumor_type_str)
                else:
                    print(f"  No actual images found in {class_path}.")

    if not all_features:
        raise ValueError(
            f"No valid features generated. Please ensure your 'data' directory (at '{DATA_ROOT_DIR}') "
            f"is correctly structured, images are valid, or set DUMMY_IMAGE_COUNT > 0 in config.py."
        )

    features_array = np.array(all_features)
    labels_array = np.array(all_labels)

    print(f"Initial loaded samples: {len(features_array)} with {features_array.shape[1]} features each.")

    # 1. Label Encoding for target variable (for classical ML)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_array)
    joblib.dump(label_encoder, LABEL_ENCODER_SAVE_PATH)
    print(f"Label encoder saved to: {LABEL_ENCODER_SAVE_PATH}")
    print(f"Class names: {label_encoder.classes_}")

    # 2. PCA for dimensionality reduction
    # Determine the number of components: min(n_samples, n_features) - 1, or a fixed number.
    # For a robust approach, we'll choose a fixed number, or a percentage of variance.
    # Let's target 95% variance or a max of 50 components, suitable for QML later.
    n_components = min(features_array.shape[0]-1, features_array.shape[1], 50) # Cap at 50 for QML
    if n_components <= 0:
        raise ValueError("Not enough samples or features for PCA. Check data loading.")

    pca = PCA(n_components=n_components) # Using an integer number of components
    features_pca = pca.fit_transform(features_array)
    joblib.dump(pca, PCA_SAVE_PATH)
    print(f"PCA model saved to: {PCA_SAVE_PATH}")
    print(f"PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Features after PCA: {features_pca.shape[1]} components.")


    # 3. Min-Max Scaling
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_pca)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"MinMaxScaler saved to: {SCALER_SAVE_PATH}")
    print(f"Features after scaling. Final shape: {features_scaled.shape}")

    return features_scaled, encoded_labels, label_encoder.classes_


def train_and_evaluate_classical_models():
    print(f"Starting Classical ML training and evaluation script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    X, y, class_names = load_and_preprocess_classical_data()

    if len(X) < 2:
        raise ValueError(f"Not enough data samples ({len(X)}) to perform train-test split. "
                        f"Please ensure enough images are available or increase `DUMMY_IMAGE_COUNT` in config.py.")
    
    # Check for class imbalance before splitting, especially for stratified split
    unique_labels, label_counts = np.unique(y, return_counts=True)
    if len(unique_labels) < 2:
        print("Warning: Only one class found in the data. Classification requires at least two classes. "
              "Model performance will be meaningless. Proceeding without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # Check if any class has only one sample
        if np.any(label_counts < 2):
            print("Warning: Some classes have only one sample. Stratified split may fail. Proceeding without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    models = {
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(random_state=42, probability=True), # probability=True for consistent API
        "KNeighborsClassifier": KNeighborsClassifier(),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=200)
    }

    results = {}
    
    os.makedirs(CLASSICAL_MODELS_DIR, exist_ok=True) # Ensure directory exists

    print("\n--- Training and Evaluating Classical Models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": cm.tolist() # Convert numpy array to list for JSON compatibility
            }

            print(f"  {name} Accuracy: {accuracy:.4f}")
            print(f"  {name} Classification Report:\n{classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")
            print(f"  {name} Confusion Matrix:\n{cm}")

            # Save the trained model
            model_save_path = os.path.join(CLASSICAL_MODELS_DIR, f'{name}_model.pkl')
            joblib.dump(model, model_save_path)
            print(f"  {name} model saved to {model_save_path}")

            # Use the shared evaluate_model function for consistency
            evaluate_model(y_test, y_pred, model_name=name, class_names=class_names, output_dir=CLASSICAL_MODELS_DIR)


        except Exception as e:
            print(f"  Error training or evaluating {name}: {e}")
            results[name] = {"error": str(e)}

    print("\n--- Classical ML Training and Evaluation Complete. ---")

if __name__ == '__main__':
    try:
        train_and_evaluate_classical_models()
    except Exception as e:
        print(f"\nAn error occurred during Classical ML training: {e}")
        import traceback
        traceback.print_exc()