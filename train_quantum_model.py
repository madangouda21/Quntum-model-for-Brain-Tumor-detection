import pennylane as qml
from pennylane import numpy as np # IMPORTANT: Use pennylane.numpy for automatic differentiation
from pennylane.optimize import AdamOptimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from datetime import datetime
from PIL import Image # For image processing in data loading

# --- Robust Imports from local package and config ---
try:
    from quantum_model.preprocessing import _extract_features_from_image
    from quantum_model.evaluation import evaluate_model
    # Import the circuit definition, not the QNode instance anymore
    from quantum_model.quantum_classifier import feature_map, ansatz, quantum_classifier_circuit, set_num_qubits # <-- CHANGED
    from config import (
        DATA_ROOT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, TUMOR_TYPES_LIST,
        PCA_SAVE_PATH, SCALER_SAVE_PATH, QML_PARAMS_SAVE_PATH, DUMMY_IMAGE_COUNT,
        QML_RESULTS_DIR, QML_QUANTUM_SCALER_SAVE_PATH
    )
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir) # Add Quantum_Computing/ to path
    sys.path.insert(0, os.path.join(current_dir, 'quantum_model')) # Add Quantum_Computing/quantum_model/ to path
    try:
        from config import (
            DATA_ROOT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, TUMOR_TYPES_LIST,
            PCA_SAVE_PATH, SCALER_SAVE_PATH, QML_PARAMS_SAVE_PATH, DUMMY_IMAGE_COUNT,
            QML_RESULTS_DIR, QML_QUANTUM_SCALER_SAVE_PATH
        )
        from preprocessing import _extract_features_from_image
        from evaluation import evaluate_model
        from quantum_classifier import feature_map, ansatz, quantum_classifier_circuit, set_num_qubits # <-- CHANGED
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import preprocessing, evaluation, quantum_classifier or config. Check sys.path and file existence. Error: {e}")
        sys.exit(1)

# Global for num_qubits, which will be set after PCA
global_num_qubits = 0

def load_and_preprocess_data_for_qml():
    """
    Loads and preprocesses data specifically for QML training.
    Applies classical PCA and a classical scaler, then a quantum-specific scaler.
    """
    print("\n--- Preparing data for Quantum ML training ---")

    all_features = []
    all_labels = [] # Will store integer labels

    # Define binary labels for QML: -1 for 'notumor', 1 for any other tumor type
    tumor_label = 1
    not_tumor_label = -1
    
    print("Mapping original classes to QML binary labels (-1 for 'notumor', 1 for others)...")

    if DUMMY_IMAGE_COUNT > 0:
        print(f"\n--- DUMMY DATA MODE ACTIVE (DUMMY_IMAGE_COUNT = {DUMMY_IMAGE_COUNT}) ---")
        print("Generating synthetic features for QML. Model performance will not be meaningful.")
        feature_vector_size = IMAGE_WIDTH * IMAGE_HEIGHT
        num_samples_per_class = DUMMY_IMAGE_COUNT // len(TUMOR_TYPES_LIST)
        
        for tumor_type_str in TUMOR_TYPES_LIST:
            binary_qml_label = not_tumor_label if tumor_type_str == 'notumor' else tumor_label
            for _ in range(num_samples_per_class):
                features = np.random.rand(feature_vector_size) # Random features
                all_features.append(features)
                all_labels.append(binary_qml_label)
        print(f"--- Generated {len(all_features)} dummy samples. ---\n")

    else: # Process actual data
        print("\n--- REAL DATA MODE ACTIVE (DUMMY_IMAGE_COUNT = 0) ---")
        print(f"Attempting to load real images from: {DATA_ROOT_DIR}")

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

                binary_qml_label = not_tumor_label if tumor_type_str == 'notumor' else tumor_label
                print(f"Loading images from: {class_path} for class: {tumor_type_str} (QML label: {binary_qml_label})")

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
                            all_labels.append(binary_qml_label)
                else:
                    print(f"  No actual images found in {class_path}.")

    if not all_features:
        raise ValueError(
            f"No valid features generated. Please ensure your 'data' directory (at '{DATA_ROOT_DIR}') "
            f"is correctly structured, images are valid, or set DUMMY_IMAGE_COUNT > 0 in config.py."
        )

    features_array_raw = np.array(all_features, requires_grad=False)
    labels_array_binary_qml = np.array(all_labels, requires_grad=False)

    print(f"Initial loaded samples: {len(features_array_raw)} with {features_array_raw.shape[1]} features each.")

    try:
        loaded_pca = joblib.load(PCA_SAVE_PATH)
        classical_scaler = joblib.load(SCALER_SAVE_PATH)
        print(f"Loaded classical PCA from: {PCA_SAVE_PATH}")
        print(f"Loaded classical MinMaxScaler from: {SCALER_SAVE_PATH}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Classical PCA or MinMaxScaler not found. "
            f"You must run 'train_classical_models.py' FIRST to generate these preprocessing objects. "
            f"Expected files at: {PCA_SAVE_PATH} and {SCALER_SAVE_PATH}"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading classical preprocessing objects: {e}")

    print("Applying classical PCA and MinMaxScaler to raw features...")
    features_array_raw_reshaped = features_array_raw.reshape(features_array_raw.shape[0], -1)
    
    features_pca = loaded_pca.transform(features_array_raw_reshaped)
    features_scaled_classical = classical_scaler.transform(features_pca)
    print(f"Features after classical PCA ({features_pca.shape[1]} components) and scaling.")

    global global_num_qubits
    global_num_qubits = features_scaled_classical.shape[1]
    set_num_qubits(global_num_qubits) # Update the global num_qubits in quantum_classifier.py
    print(f"Number of qubits (PCA components) set to: {global_num_qubits}")

    print("Applying quantum-specific MinMaxScaler (range [0, pi])...")
    quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
    features_quantum_scaled = quantum_scaler.fit_transform(features_scaled_classical)
    
    joblib.dump(quantum_scaler, QML_QUANTUM_SCALER_SAVE_PATH)
    print(f"Quantum MinMaxScaler saved to: {QML_QUANTUM_SCALER_SAVE_PATH}")

    print("Features ready for QML training.")

    unique_labels, counts = np.unique(labels_array_binary_qml, return_counts=True)
    label_info = []
    for label, count in zip(unique_labels, counts):
        if label == -1:
            label_info.append(f"No Tumor (-1): {count}")
        elif label == 1:
            label_info.append(f"Tumor (1): {count}")
        else:
            label_info.append(f"Unknown Label ({label}): {count}")
    print(f"Binary label counts (for QML -1/1): {', '.join(label_info)}")


    return features_quantum_scaled, labels_array_binary_qml


def train_and_evaluate_quantum_model():
    print(f"Starting Quantum ML training and evaluation script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X, y_binary = load_and_preprocess_data_for_qml()

    if len(X) < 2:
        raise ValueError(f"Not enough data samples ({len(X)}) to perform train-test split for QML. "
                        f"Please ensure enough images are available or increase `DUMMY_IMAGE_COUNT` in config.py.")

    if not np.all(np.isin(y_binary, [-1, 1])):
        raise ValueError("QML labels must be binary (-1 or 1). Please check data loading logic.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"QML Training samples: {len(X_train)}, QML Test samples: {len(X_test)}")

    # Define the number of layers for the ansatz
    num_layers = 6 # Set your desired number of layers here

    # --- CHANGE STARTS HERE ---
    # Dynamically create the device and QNode AFTER global_num_qubits is known
    dev = qml.device("lightning.qubit", wires=global_num_qubits)

    @qml.qnode(dev)
    def qml_model(x, params, num_ansatz_layers): # <-- This is your QNode instance now
        return quantum_classifier_circuit(x, params, num_ansatz_layers)
    # --- CHANGE ENDS HERE ---

    # Initialize quantum model parameters:
    params_size_for_ansatz = num_layers * global_num_qubits
    params = np.random.uniform(low=-np.pi, high=np.pi, size=params_size_for_ansatz, requires_grad=True)

    print(f"Initialized QML parameters with shape: {params.shape} (total size {params.size})")

    opt = AdamOptimizer(stepsize=0.01)
    batch_size = 32
    steps = 100 # Number of optimization steps

    print(f"Starting QML training for {steps} steps with batch size {batch_size}...")
    
    cost_history = []
    
    def hinge_loss(predictions, true_labels):
        return np.mean(np.maximum(0, 1 - true_labels * predictions))

    for i in range(steps):
        batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        def cost_fn(params, features, labels):
            predictions = np.array([qml_model(f, params, num_layers) for f in features])
            return hinge_loss(predictions, labels)

        params, cost_value = opt.step_and_cost(lambda p: cost_fn(p, X_batch, y_batch), params)
        cost_history.append(cost_value)

        if (i + 1) % 10 == 0:
            print(f"Step {i + 1} | Cost: {cost_value:.4f}")

    print("\nQML training complete.")
    
    qml_params_save_path = QML_PARAMS_SAVE_PATH
    joblib.dump(params, qml_params_save_path)
    print(f"Trained QML parameters saved to: {qml_params_save_path}")

    print("\nEvaluating QML model on test set...")
    y_pred_raw = np.array([qml_model(x, params, num_layers) for x in X_test])
    
    y_pred_binary = np.where(y_pred_raw >= 0.0, 1, -1)

    qml_class_names = ['notumor', 'tumor_detected'] 
    
    y_test_eval = np.where(y_test == -1, 0, 1)
    y_pred_binary_eval = np.where(y_pred_binary == -1, 0, 1)

    evaluate_model(y_test_eval, y_pred_binary_eval, model_name="Quantum_Model", class_names=qml_class_names, output_dir=QML_RESULTS_DIR)

    print("\n--- Quantum ML Training and Evaluation Complete. ---")

if __name__ == '__main__':
    try:
        train_and_evaluate_quantum_model()
    except Exception as e:
        print(f"\nAn error occurred during Quantum ML training: {e}")
        import traceback
        traceback.print_exc()