# Quntum-model-for-Brain-Tumor-detection

Motivation

Brain tumors are life-threatening conditions that require early and accurate diagnosis. Manual analysis of MRI scans is time-consuming and error-prone. This project explores how the power of machine learning combined with quantum computing can automate and improve tumor detection — aiming for faster, smarter, and more reliable healthcare solutions.

Quantum Model for Brain Tumor Detection

This project leverages classical machine learning combined with quantum computing concepts to detect brain tumors from MRI images.

Setup Instructions

Create Python Environment

python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Dataset

You need to download the Brain Tumor MRI dataset from Kaggle:

Dataset Link: Brain Tumor Classification (MRI)

After downloading, extract and place the dataset inside the project root folder:

Quntum-model-for-Brain-Tumor-detection/Dataset/

Run the Project

Step 1: Feature Extraction

Extract features from the MRI images and save them in a .csv file.

python feature_extraction.py

This will generate a CSV file with all the extracted features used for training the model.
python train_classical_models.py   // for comparing the models both in once
Step 2: Train the Model

python train_model.py

You can switch between classical and quantum classifiers inside the script.

Technologies Used
	•	Python
	•	Scikit-learn
	•	NumPy, Pandas
	•	Qiskit / PennyLane (for quantum models)
	•	Matplotlib & Seaborn (for visualization)


📌 Notes
	•	Ensure your system has proper packages installed for running Qiskit (or Pennylane if you’re using it).
	•	You may require Jupyter or Google Colab if you face local runtime issues for quantum models.
	•	The code is modular: feature extraction, training, and evaluation are all in separate scripts.



i uploaded app.py because to connect frontend , the frontend code you can downloaded from my another Repository
