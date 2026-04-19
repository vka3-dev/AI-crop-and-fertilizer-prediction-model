# 🌾 Smart Crop & Fertilizer Recommendation

![App Screenshot](https://github.com/srinitish/Smart-Crop-Recommendation-System-with-Fertilizer-Suggestion/blob/master/app_img.png?raw=true)

A machine learning–powered web app that recommends the **best crop** and a matching **fertilizer** based on soil and climate conditions. Built for Indian agriculture and deployed with a clean, interactive Streamlit UI.

---

## 🚀 What the App Does

1. **Crop prediction (Stage 1)**
   - Inputs: `N`, `P`, `K`, `temperature`, `humidity`, `pH`, `rainfall`
   - Model: **Logistic Regression** inside a scikit‑learn **Pipeline**
   - Steps:
     - Standardization with `StandardScaler`
     - `LabelEncoder` for crop names
     - Hyperparameter tuning with `GridSearchCV`
   - Output: Recommended crop (e.g. rice, maize, pulses, fruits, etc.)

2. **Fertilizer prediction (Stage 2)**
   - Inputs:  
     - Shared: `Temperature`, `Rainfall`, `pH`, `Nitrogen`, `Phosphorous`, `Potassium`  
     - Extra: `Moisture`, `Carbon`, encoded `Soil` type, encoded `Crop` from Stage 1
   - Model: **RandomForestClassifier** with a tuned parameter grid
   - Steps:
     - Separate `LabelEncoder` for fertilizer names
     - Encoders saved and reused with `joblib`
     - Class‑imbalance handling (class weights / resampling)
   - Output: Recommended fertilizer (e.g. Urea, DAP, MOP, compost, balanced NPK, etc.)

3. **End‑to‑end pipeline**
   - User fills a **single form** with field details.
   - App:
     - Passes features to the crop pipeline → predicts crop.
     - Re‑encodes the predicted crop into the fertilizer feature space.
     - Predicts the best fertilizer using the Random Forest pipeline.
   - All models and encoders are loaded from `.plk` files:
     - `crop_model.plk`, `crop_encode.plk`
     - `fertilizer.plk`, `fertilizer_encode.plk`
     - `soil_encode.plk`, `ferti_crop_encode.plk`

---

## 📊 Data & Analysis

- Crop dataset with 20+ crops (fruits, cereals, pulses, cash crops).
- Fertilizer dataset with multiple fertilizer types and soil attributes.
- Performed **EDA** with:
  - Count plots for crop and fertilizer distributions
  - Box/violin plots for feature distributions per class
  - Correlation heatmaps and IQR‑based outlier removal
  - PCA visualizations to inspect class separability
- Evaluated models using accuracy, macro F1, confusion matrices and classification reports.

---

## 🧠 Tech Stack

- **Python**, **NumPy**, **Pandas**
- **scikit‑learn**: LogisticRegression, RandomForestClassifier, Pipeline, SelectKBest, StandardScaler, GridSearchCV, LabelEncoder
- **Streamlit**: interactive UI for real‑time recommendations
- **Matplotlib / Seaborn**: visualizations for EDA and reports
- **joblib**: saving/loading trained models and encoders

---

## 💡 How It Can Be Used

- As a **decision support tool** to suggest suitable crop–fertilizer pairs.
- As a **learning project** demonstrating:
  - Multi‑stage ML pipelines
  - Encoders shared across models
  - Model deployment with Streamlit
  - Handling imbalanced, real‑world agricultural data.
