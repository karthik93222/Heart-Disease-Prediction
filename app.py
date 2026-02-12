from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from waitress import serve

app = Flask(__name__)

# Paths to pickles
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

def load_artifacts():
    """Load model, scaler, and label encoder if available."""
    model = scaler = le = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
    return model, scaler, le

@app.route('/', methods=['GET'])
def home():
    # Render form for input features
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model, scaler, le = load_artifacts()
    if model is None:
        return "Model file not found. Please create `model.pkl` in the project directory.", 500

    # Expected feature order (same as training):
    fields = [
        'slope_of_peak_exercise_st_segment',
        'thal',
        'resting_blood_pressure',
        'chest_pain_type',
        'num_major_vessels',
        'fasting_blood_sugar_gt_120_mg_per_dl',
        'resting_ekg_results',
        'serum_cholesterol_mg_per_dl',
        'oldpeak_eq_st_depression',
        'sex',
        'age',
        'max_heart_rate_achieved',
        'exercise_induced_angina'
    ]

    # Collect inputs from form
    values = []
    for f in fields:
        val = request.form.get(f)
        if val is None:
            return f"Missing form field: {f}", 400
        values.append(val)

    # Convert types: 'thal' may be categorical string and must be encoded
    # Create numpy array with proper numeric types
    try:
        # Handle 'thal' separately if label encoder present
        if le is not None:
            thal_val = le.transform([values[1]])[0]
        else:
            # If no label encoder, try to use value as-is (maybe numeric already)
            try:
                thal_val = float(values[1])
            except Exception:
                # fallback: use raw string won't work with scaler; raise error
                return "Label encoder not found and 'thal' is non-numeric. Save the label encoder or provide numeric thal value.", 500

        # Build numeric list replacing thal
        numeric_vals = []
        for i, v in enumerate(values):
            if i == 1:
                numeric_vals.append(thal_val)
                continue
            # Most other fields are numeric in the dataset
            numeric_vals.append(float(v))

        X = np.array(numeric_vals).reshape(1, -1)

        # Apply scaler if available
        if scaler is not None:
            X = scaler.transform(X)

        pred_proba = None
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]

        label_text = 'Heart disease' if int(pred) == 1 else 'No heart disease'

        return render_template(
            'index.html',
            prediction=int(pred),
            label=label_text,
            probability=round(float(pred_proba) if pred_proba is not None else None, 4),
            inputs=dict(zip(fields, values)),
        )

    except Exception as e:
        return f"Error during prediction: {e}", 500

if __name__ == '__main__':
    #serve(app, host='0.0.0.0', port=8080)
    app.run(debug=True)