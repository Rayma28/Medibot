# app.py - Main Flask application

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.static_folder = 'static'

# Load data and models
def load_data():
    data = pd.read_csv('data/disease_symptoms.csv')
    # Print column names for debugging
    print("CSV columns:", data.columns.tolist())
    return data

def check_models():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Check if models exist, if not train them
    if not (os.path.exists('models/rf_model.pkl') and os.path.exists('models/dl_model.h5')):
        print("Models not found, training now...")
        train_models()
    
    # Load the trained models
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    dl_model = load_model('models/dl_model.h5')
    
    # Load encoders and symptom list
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open('models/symptoms.json', 'r') as f:
        all_symptoms = json.load(f)
    
    return rf_model, dl_model, label_encoder, all_symptoms

def train_models():
    # Check if dataset exists, if not, create a sample dataset
    if not os.path.exists('data/disease_symptoms.csv'):
        print("Creating sample dataset...")
        create_sample_dataset()
    
    # Load the dataset
    data = load_data()
    
    # Get all symptom columns (all columns except 'Disease')
    symptom_columns = [col for col in data.columns if col != 'Disease']
    all_symptoms = symptom_columns
    
    # Save symptoms list
    with open('models/symptoms.json', 'w') as f:
        json.dump(all_symptoms, f)
    
    # Prepare features (X) and target (y)
    X = data[symptom_columns].values
    y = data['Disease'].values
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save the label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y_encoded)
    
    # Save the Random Forest model
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Prepare data for Deep Learning model
    y_categorical = to_categorical(y_encoded)
    
    # Build Deep Learning model
    dl_model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(y_categorical.shape[1], activation='softmax')
    ])
    
    dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train Deep Learning model
    dl_model.fit(X, y_categorical, epochs=50, batch_size=32, verbose=0)
    
    # Save the Deep Learning model
    dl_model.save('models/dl_model.h5')
    
    print("Models trained and saved successfully!")

def create_sample_dataset():
    """Create a sample disease-symptoms dataset in the binary format"""
    sample_data = {
        'Fever': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'Cough': [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
        'Fatigue': [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        'Headache': [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        'Sore Throat': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Disease': [
            'Common Cold',
            'Influenza',
            'Pneumonia',
            'Diabetes',
            'Hypertension',
            'Asthma',
            'Migraine',
            'Arthritis',
            'Gastritis',
            'Bronchitis'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/disease_symptoms.csv', index=False)
    print("Sample dataset created!")

# Initialize models
rf_model, dl_model, label_encoder, all_symptoms = check_models()

def predict_disease(symptoms_text):
    # Process the input symptoms
    input_symptoms = [symptom.strip().lower() for symptom in symptoms_text.split(',')]
    
    # Create feature vector
    X = np.zeros(len(all_symptoms))
    found_symptoms = []
    
    for symptom in input_symptoms:
        for i, known_symptom in enumerate(all_symptoms):
            if symptom in known_symptom.lower():
                X[i] = 1
                found_symptoms.append(known_symptom)
    
    if not found_symptoms:
        return {
            "status": "error",
            "message": "No recognized symptoms found. Please describe your symptoms more clearly."
        }
    
    # Make predictions with both models
    rf_pred = rf_model.predict_proba([X])[0]
    dl_pred = dl_model.predict([X.reshape(1, -1)])[0]
    
    # Get top 3 predictions from each model
    rf_top3_indices = rf_pred.argsort()[-3:][::-1]
    dl_top3_indices = dl_pred.argsort()[-3:][::-1]
    
    rf_diseases = [label_encoder.inverse_transform([idx])[0] for idx in rf_top3_indices]
    rf_probabilities = [float(rf_pred[idx] * 100) for idx in rf_top3_indices]
    
    dl_diseases = [label_encoder.inverse_transform([idx])[0] for idx in dl_top3_indices]
    dl_probabilities = [float(dl_pred[idx] * 100) for idx in dl_top3_indices]
    
    # Combine results from both models
    combined_results = {}
    
    # Weight RF model results (60%)
    for disease, prob in zip(rf_diseases, rf_probabilities):
        combined_results[disease] = 0.6 * prob
    
    # Weight DL model results (40%)
    for disease, prob in zip(dl_diseases, dl_probabilities):
        if disease in combined_results:
            combined_results[disease] += 0.4 * prob
        else:
            combined_results[disease] = 0.4 * prob
    
    # Sort combined results
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    
    # Format results
    predictions = []
    for disease, probability in sorted_results[:3]:
        predictions.append({
            "disease": disease,
            "probability": round(probability, 2)
        })
    
    return {
        "status": "success",
        "identified_symptoms": found_symptoms,
        "predictions": predictions,
        "disclaimer": "This is not a substitute for professional medical advice. Please consult a doctor for proper diagnosis."
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_symptoms')
def get_symptoms():
    return jsonify(all_symptoms)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    
    if not symptoms:
        return jsonify({
            "status": "error",
            "message": "No symptoms provided"
        })
    
    result = predict_disease(symptoms)
    return jsonify(result)

if __name__ == '__main__':
    # If the dataset already exists but has the wrong columns, delete it to recreate
    if os.path.exists('data/disease_symptoms.csv'):
        try:
            df = pd.read_csv('data/disease_symptoms.csv')
            if 'Disease' not in df.columns:
                print("Existing dataset has incorrect columns. Recreating...")
                os.remove('data/disease_symptoms.csv')
        except:
            print("Error reading existing dataset. Recreating...")
            os.remove('data/disease_symptoms.csv')
    
    # Delete existing models to force retraining with correct data
    for model_file in ['models/rf_model.pkl', 'models/dl_model.h5', 'models/label_encoder.pkl', 'models/symptoms.json']:
        if os.path.exists(model_file):
            os.remove(model_file)
    
    app.run(debug=True)