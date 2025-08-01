# Quantum_Computing/main.py

import os
import sys

# Add the Quantum_Computing directory to the Python path if running from parent
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def run_classical_training():
    print("\n--- Running Classical Model Training ---")
    try:
        from train_classical_models import train_and_evaluate_classical_models
        train_and_evaluate_classical_models()
        print("Classical training completed successfully.")
    except Exception as e:
        print(f"Error during classical training: {e}")
        import traceback
        traceback.print_exc()

def run_quantum_training():
    print("\n--- Running Quantum Model Training ---")
    try:
        from train_quantum_model import train_and_evaluate_quantum_model
        train_and_evaluate_quantum_model()
        print("Quantum training completed successfully.")
    except Exception as e:
        print(f"Error during quantum training: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting Main Application Workflow...")
    
    # Step 1: Train Classical Models (generates PCA and MinMaxScaler needed for QML)
    run_classical_training()

    # Step 2: Train Quantum Model (requires classical preprocessing artifacts)
    run_quantum_training()

    print("\nMain Application Workflow Completed.")

if __name__ == '__main__':
    main()