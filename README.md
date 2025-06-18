# 🏭 Factory Emission Compliance Predictor

### 🔬 Built by Team Innov8ers | Theme: Open Innovation

An AI-powered Streamlit web app that predicts whether a factory is **compliant or non-compliant** with pollution control norms based on emissions and factory conditions. Designed for scalable, automated, and explainable industrial monitoring.

---

## 📌 Problem Statement

Manual monitoring of industrial emissions is slow and time consuming. Many factories exceed pollution limits for gases like **SOx**, **NOx**, and **CO₂**. A smarter, real-time solution is needed to help authorities prioritize inspections and maintain environmental safety.

---

## 🎯 Project Objective

Develop a machine learning-powered web tool that:
- Accepts factory emission data via form or CSV
- Predicts compliance status (Compliant / Non-compliant)
- Explains reasons behind non-compliance
- Visualizes emission data using charts and maps

---

## 🛠️ Tech Stack

| Layer       | Tool/Library              |
|-------------|---------------------------|
| Frontend    | Streamlit                 |
| Backend     | Python, Pandas, NumPy     |
| ML Model    | Scikit-learn (Random Forest) |
| Visualization | Altair, Pydeck         |
| Hosting     | Streamlit Cloud           |

---

## 📥 Input Features

- `Factory Name`
- `Industry Type` (e.g., Textile, Steel, Cement)
- `SOx` (ppm)
- `NOx` (ppm)
- `CO₂` (ppm)
- `Scrubber Efficiency (%)`
- `Production Volume (tons/month)`
- `Plant Age (years)`

---

## 💻 How to Run Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the Streamlit app
streamlit run app.py
