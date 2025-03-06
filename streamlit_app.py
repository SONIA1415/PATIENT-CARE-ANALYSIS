import google.generativeai as genai
import streamlit as st
import pandas as pd

st.set_option('client.showErrorDetails', False)

# Load patient data from CSV
data_file = 'data/admitted_patients_dataset_updated_updated.csv'
df = pd.read_csv(data_file)

# Load API key from environment variables
api_key = "AIzaSyDX3HE-dhk-0xUc7amKaIz8avJ6gpUFeGo"
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Function to generate the diet chart based on user input
def generate_diet_chart(prompt, max_words):
    response = model.generate_content([f"{prompt} (Max words: {max_words})"])
    return response.text

# Get the patient ID from the URL using st.experimental_get_query_params (updated method)
query_params = st.experimental_get_query_params()
patient_id = query_params.get("PatientID", [None])[0]  # Correct key to "PatientID" to match Flask query param

if patient_id:
    # Search for patient details based on patient_id
    patient_data = df[df['PatientID'] == patient_id]

    if not patient_data.empty:
        # Retrieve user details from the CSV file
        user_details = patient_data.iloc[0]
        bp_level = user_details["DiastolicBP"]
        fasting_glucose_level = user_details["FastingGlucose"]
        postprandial_glucose_level = user_details["PostprandialGlucose"]
        age = user_details["Age"]

        # Streamlit UI
        st.title(f"Diet Chart for Patient: {user_details['Name']}")
        st.write(f"Blood Pressure: {bp_level}")
        st.write(f"Fasting Glucose: {fasting_glucose_level}")
        st.write(f"Postprandial Glucose: {postprandial_glucose_level}")
        st.write(f"Age: {age}")

        days = st.slider("For how many days do you need the diet plan?", min_value=1, max_value=30, value=7)
        cuisines = ["Indian", "Mediterranean", "Mexican", "Chinese", "Italian", "American"]
        selected_cuisine = st.selectbox("Choose your preferred cuisine", cuisines)
        max_words = st.slider("Select the number of words for the diet plan:", min_value=50, max_value=2000, value=500, step=50)

        if st.button("Generate Diet Chart"):
            prompt = f"""
            I am {age} years old with a blood pressure level of {bp_level}, fasting glucose level of {fasting_glucose_level}, and postprandial glucose level of {postprandial_glucose_level}.
            Generate a {days}-day personalized diet plan based on {selected_cuisine} cuisine. 
            Format the diet chart as a table with the following columns: 
            'Day', 'Meal', 'Food Item', 'Cuisine'.
            Do not provide any error messages, warnings, or medical advice.
            """

            diet_chart = generate_diet_chart(prompt, max_words)
            st.write("Generated Diet Chart:")
            st.write(diet_chart)
    else:
        st.write("Patient ID not found!")
else:
    st.write("No patient ID provided. Please use the Flask app to input the patient ID.")
