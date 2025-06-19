import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk
import altair as alt
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate data again (or load if available)
def generate_factory_data(n=1000):
    industry_types = ["Textile", "Chemical", "Steel", "Cement", "Pharmaceutical"]
    data = []
    for _ in range(n):
        industry = np.random.choice(industry_types)
        sox = np.round(np.random.uniform(50, 600), 2)
        nox = np.round(np.random.uniform(30, 500), 2)
        co2 = np.round(np.random.uniform(100, 1000), 2)
        volume = np.round(np.random.uniform(100, 5000), 2)
        scrub = np.round(np.random.uniform(40, 100), 2)
        age = np.random.randint(1, 51)
        lat = np.random.uniform(18.4, 19.0)  # Random lat for map demo
        lon = np.random.uniform(73.7, 74.2)  # Random lon for map demo
        if (sox > 400 or nox > 300 or co2 > 800) and scrub < 70:
            compliance = 1  # Non-compliant
        else:
            compliance = 0  # Compliant
        data.append([industry, sox, nox, co2, volume, scrub, age, lat, lon, compliance])
    return pd.DataFrame(data, columns=["industry_type", "sox_ppm", "nox_ppm", "co2_ppm", "production_volume", "scrubber_efficiency", "plant_age", "lat", "lon", "compliance"])

# Prepare data
df = generate_factory_data()
le = LabelEncoder()
df['industry_encoded'] = le.fit_transform(df['industry_type'])

# Features and target
X = df[['industry_encoded', 'sox_ppm', 'nox_ppm', 'co2_ppm', 'production_volume', 'scrubber_efficiency', 'plant_age']]
y = df['compliance']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏭 Factory Emission Compliance Predictor")
st.markdown("Enter factory details below or upload a CSV to check compliance.")

# User input section
factory_name = st.text_input("Factory Name")
email = st.text_input("Email Address to Send Report")
industry = st.selectbox("Industry Type", ["Textile", "Chemical", "Steel", "Cement", "Pharmaceutical"])
sox = st.slider("SOx level (ppm)", 50, 600, 200)
nox = st.slider("NOx level (ppm)", 30, 500, 150)
co2 = st.slider("CO₂ level (ppm)", 100, 1000, 400)
volume = st.number_input("Production Volume (tons/month)", 100.0, 5000.0, 1200.0)
scrub = st.slider("Scrubber Efficiency (%)", 40, 100, 75)
age = st.slider("Plant Age (years)", 1, 50, 15)

# Email report function
def send_email_report(to_email, subject, body):
    try:
        sender = "factoryemissionreport@gmail.com"
        password = "factoryemit12*"

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email failed to send: {e}")
        return False

if st.button("Predict Compliance"):
    industry_encoded = le.transform([industry])[0]
    input_data = np.array([[industry_encoded, sox, nox, co2, volume, scrub, age]])
    pred = model.predict(input_data)[0]
    result = "✅ Compliant" if pred == 0 else "❌ Non-compliant"
    st.subheader(f"Prediction for {factory_name or 'this factory'}: {result}")
    proba = model.predict_proba(input_data)[0][pred]
    st.caption(f"Model confidence: {proba*100:.2f}%")

    # Simple explanation
    reasons = []
    if sox > 400: reasons.append("High SOx")
    if nox > 300: reasons.append("High NOx")
    if co2 > 800: reasons.append("High CO₂")
    if scrub < 70: reasons.append("Low scrubber efficiency")

    suggestions = [
        "Ensure scrubber maintenance and efficiency above 70%.",
        "Consider upgrading filtration equipment.",
        "Regularly monitor and log emission levels.",
    ]

    if pred == 1:
        st.markdown("**Reason(s) for Non-compliance:**")
        for r in reasons:
            st.markdown(f"- {r}")

    # Send report to email
    if email:
        email_body = f"Factory Name: {factory_name or 'N/A'}\nResult: {result}\n\nReasons:\n- " + "\n- ".join(reasons)
        if pred == 1:
            email_body += "\n\nSuggestions:\n- " + "\n- ".join(suggestions)
        sent = send_email_report(email, f"Emission Compliance Report: {factory_name or 'Factory'}", email_body)
        if sent:
            st.success(f"Report sent to {email}")

# CSV Upload for batch prediction
st.header("📥 Upload CSV for Batch Prediction")
csv = st.file_uploader("Upload a CSV with factory data", type=["csv"])
if csv:
    csv_df = pd.read_csv(csv)
    if 'industry_type' in csv_df.columns:
        csv_df['industry_encoded'] = le.transform(csv_df['industry_type'])
        features = csv_df[['industry_encoded', 'sox_ppm', 'nox_ppm', 'co2_ppm', 'production_volume', 'scrubber_efficiency', 'plant_age']]
        preds = model.predict(features)
        csv_df['prediction'] = np.where(preds == 0, "✅ Compliant", "❌ Non-compliant")
        st.write(csv_df[['industry_type', 'sox_ppm', 'nox_ppm', 'co2_ppm', 'scrubber_efficiency', 'plant_age', 'prediction']])
        st.download_button("Download Results", csv_df.to_csv(index=False), "predictions.csv")
    else:
        st.warning("CSV must include 'industry_type', 'sox_ppm', 'nox_ppm', 'co2_ppm', 'production_volume', 'scrubber_efficiency', 'plant_age'")

# Visual Analytics
st.header("📊 Visual Analytics")
# Compliance pie chart
compliance_chart = pd.DataFrame(df['compliance'].map({0: 'Compliant', 1: 'Non-compliant'}).value_counts()).reset_index()
compliance_chart.columns = ['Status', 'Count']
st.altair_chart(alt.Chart(compliance_chart).mark_bar().encode(
    x='Status',
    y='Count',
    color='Status'
), use_container_width=True)

# Map view
st.header("🗺️ Factory Map View")
st.map(df[['lat', 'lon']])

# Optional: show training accuracy
if st.checkbox("Show model training accuracy"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy on Test Data: {acc*100:.2f}%")
