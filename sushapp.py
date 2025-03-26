import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load provided dataset
df = pd.read_csv('healthcare_customer_loyalty.csv')

# Preprocessing
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


X = df.drop('Loyalty', axis=1)
y = df['Loyalty']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train with Random Forest (chosen for high accuracy)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Evaluation
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(rf, 'hospital_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# 2. Streamlit Website Code (Save as app.py)
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('hospital_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏥 Hospital Management & Patient Loyalty Prediction")

st.write("Enter the following details:")

# Auto-generate input fields based on feature columns
feature_names = [col for col in df.columns if col != 'Loyalty_Score']

user_input = {}
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=1.0)
    user_input[feature] = value

if st.button("Predict Loyalty Score"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Loyalty Score: {prediction[0]}")

st.sidebar.header("About")
st.sidebar.info("This app predicts customer loyalty scores for healthcare customers based on provided data.")

