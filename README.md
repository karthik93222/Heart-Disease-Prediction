# Heart Disease Prediction — Flask App

This project contains a Jupyter notebook that trains multiple models and a small Flask web app to make predictions from user input.

Files of interest
- `PRCP-1016-HeartDieseasePred.ipynb` — exploratory analysis, preprocessing, model training, and (new) pipeline-saving cell that writes `model.pkl`.
- `app.py` — Flask application that loads the saved pipeline (if present) and exposes endpoints for prediction and retraining.
- `heart.csv` — dataset used for training and for reconstructing encoders/feature order.
- `templates/index.html` — web form for entering patient data and getting prediction results.

Quick setup (Windows PowerShell)
1. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies
```powershell
pip install -r requirements.txt
```

Run the web app
```powershell
python app.py
```
Open http://127.0.0.1:5000/ in your browser.

Training and model files
- The notebook saves two different artifact styles (you may have one or both saved):
  - Separate artifacts: `gradient_boosting_model_joblib.pkl` and `scaler_joblib.pkl` (these are created by cells in `final.ipynb`). If you prefer the app to use separate artifacts, see the notebook cells and the `app.py` adjustments.
  - Combined pipeline: `model.pkl` — created by the added notebook cell which trains a full preprocessing + SMOTE + GradientBoosting pipeline and saves it. The Flask app will load this combined pipeline when present.

How the Flask app uses the model
- At startup the app tries to load `model.pkl`. If present, it uses that pipeline (recommended).
- If not present, use the notebook to create `model.pkl` by running the added cell (or use the `/train` endpoint in the app which trains and saves the pipeline).

Endpoints
- `/` (GET) — main page with prediction form.
- `/predict` (POST) — accepts form data and returns prediction (renders same `index.html` with results).
- `/train` (GET/POST) — small UI to retrain the pipeline using `heart.csv` and overwrite `model.pkl`.

Important notes
- Security: `/train` triggers training on the server and should not be exposed publicly without authentication.
- Reproducibility: saving the LabelEncoder(s) or the full pipeline from the notebook ensures exact encoding at prediction time. The notebook now includes a cell that saves `model.pkl`.
- Blocking: retraining runs synchronously in the Flask request — it will block the server while training. For production, use an async/background job queue or run training offline.

Troubleshooting
- If you see errors about missing files, ensure `heart.csv`, `gradient_boosting_model_joblib.pkl` (if relying on separate artifacts), or `model.pkl` exist in the project root.
- If `thal` or other categorical input values cause mapping errors, run the notebook cell that re-creates and saves the pipeline, or save and load the same encoders from the notebook.

Next steps (suggested)
- Option A: Run the notebook cell that trains & saves `model.pkl`, then start the Flask app and test predictions.
- Option B: If you want to keep separate `scaler_joblib.pkl` + `gradient_boosting_model_joblib.pkl`, tell me and I will patch `app.py` to load those artifacts and reconstruct encoders from `heart.csv`.

If you'd like, I can run the notebook cell here to create `model.pkl` and then test a sample prediction through the Flask app.
