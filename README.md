# Mammography Triage Prototype — Coursework

Student coursework prototype. NOT a medical device; never use for
medical decisions.

## What it does
Classifies a mammography image as BENIGN or MALIGNANT with a confidence
score, using a transfer-learning CNN (MobileNetV2 backbone) trained on
the CBIS-DDSM dataset (Kaggle mirror). Full documentation, evaluation
and limitations are in the companion Jupyter notebook.

## Run locally
1. `python -m venv venv` and activate it
2. `pip install -r requirements.txt`
3. Place `mammography_model.keras` in this folder
4. `streamlit run app.py`

## Model file
`mammography_model.keras` — Keras v3 format, saved with TensorFlow 2.20.0.
Preprocessing is built into the model (resize to 224×224, RGB, internal
normalisation to [-1, +1]) — no extra preprocessing is needed in the app.
