from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash,abort
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import csv
import os
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Paths to CSV files
DATA_FILE = 'data/admitted_patients_dataset_updated_updated.csv'
DOCTOR_CSV = 'data/doctors.csv'
NURSE_CSV = 'data/nurses.csv'
ADMIN_CSV = 'data/Admin.csv'
PATIENT_CSV = 'data/patients.csv'

# Load the Random Forest model and scaler
MODEL_PATH = 'models/random_forest_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

load_model_and_scaler()

# Load data using Pandas
df = pd.read_csv(DATA_FILE)
doctors_df = pd.read_csv(DOCTOR_CSV)

def read_csv(file_path):
    """Helper function to read CSV data."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return list(csv.DictReader(file))
    return []

def read_admin_credentials():
    """Load admin credentials from CSV."""
    admin_credentials = {}
    if os.path.exists(ADMIN_CSV):
        with open(ADMIN_CSV, mode='r') as file:
            reader = csv.DictReader(file)
            admin_credentials = {row['AdminID'].strip(): row['Password'].strip() for row in reader}
    return admin_credentials

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home2')
def home2():
    return render_template('Home2.html')

@app.route('/admin-log', methods=['GET', 'POST'])
def admin_log():
    if request.method == 'POST':
        admin_id = request.form.get('AdminID').strip()
        password = request.form.get('Password').strip()
        admin_credentials = read_admin_credentials()

        if admin_id in admin_credentials and admin_credentials[admin_id] == password:
            return redirect(url_for('admin'))
        flash('Invalid Admin ID or Password')
    return render_template('admin-log.html')

@app.route('/admin')
def admin():
    return render_template('Admin.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/register-doctor', methods=['POST'])
def register_doctor():
    try:
        doctor_data = {
            'DoctorID': request.form['DoctorID'],
            'DoctorName': request.form['DoctorName'],
            'PhoneNumber': request.form['PhoneNumber'],
            'Password': request.form['Password'],
            'Address': request.form['Address'],
            'SpecializationIn': request.form['SpecializationIn'],
            'JoiningDate': request.form['JoiningDate'],
            'Gender': request.form['Gender']
        }

        file_exists = os.path.isfile(DOCTOR_CSV)
        with open(DOCTOR_CSV, mode='a', newline='') as csv_file:
            fieldnames = ['DoctorID', 'DoctorName', 'PhoneNumber', 'Password', 'Address', 'SpecializationIn', 'JoiningDate', 'Gender']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(doctor_data)

        return jsonify({'success': True, 'message': 'Doctor registered successfully.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/doctor-nurse-login', methods=['GET', 'POST'])
def doctor_nurse_login():
    if request.method == 'POST':
        user_type = request.form['user_type']
        user_id = request.form['user_id']
        password = request.form['password']

        users_csv = DOCTOR_CSV if user_type == 'doctor' else NURSE_CSV
        user_id_field = 'DoctorID' if user_type == 'doctor' else 'NurseID'

        users = read_csv(users_csv)
        user = next((user for user in users if user[user_id_field] == user_id and user['Password'] == password), None)
        if user:
            session['user_data'] = user
            return redirect(url_for(f'{user_type}_profile', **{f'{user_type}_id': user_id}))
        flash(f'Invalid {user_type.capitalize()} ID or Password')
    
    return render_template('doctor_nurse_login.html')

@app.route('/doctor_profile/<doctor_id>')
def doctor_profile(doctor_id):
    doctor_details = doctors_df[doctors_df['DoctorID'] == doctor_id]
    if not doctor_details.empty:
        doctor_data = doctor_details.iloc[0].to_dict()
        return render_template('doctor_pro.html', doctor=doctor_data)
    flash('Doctor not found or does not exist.')
    return redirect(url_for('doctor_nurse_login'))

@app.route('/doctor_pro')
def doctor_pro():
    if 'user_data' in session:
        return render_template('doctor_pro.html', doctor=session['user_data'])
    return redirect(url_for('doctor_nurse_login'))

@app.route('/nurse_profile/<nurse_id>')
def nurse_profile(nurse_id):
    nurses_df = pd.read_csv(NURSE_CSV)
    nurse_details = nurses_df[nurses_df['NurseID'] == nurse_id].iloc[0]
    return render_template('Nurse.html', nurse=nurse_details)

@app.route('/Nurse')
def Nurse():
    if 'user_data' in session:
        return render_template('Nurse.html', nurse=session['user_data'])
    return redirect(url_for('doctor_nurse_login'))

@app.route('/login1', methods=['GET', 'POST'])
def login1():
    if request.method == 'POST':
        patient_id = request.form.get('PatientID').strip()
        contact_details = request.form.get('ContactDetails').strip()
        patient_data = df[(df['PatientID'].astype(str).str.strip() == patient_id) & 
                          (df['ContactDetails'].astype(str).str.strip() == contact_details)]

        if not patient_data.empty:
            session['patient_data'] = patient_data.iloc[0].to_dict()
            return redirect(url_for('profile'))
        flash('Invalid Patient ID or Contact Number')
    
    return render_template('Login1.html')

@app.route('/profile')
def profile():
    if 'patient_data' in session:
        return render_template('profile.html', **session['patient_data'])
    return redirect(url_for('login1'))

@app.route('/dashboard_view', methods=['GET'])
def dashboard_view():
    # Get the patient data from the session
    patient = session.get('patient_data')

    if not patient:
        abort(404, description="Patient not found in session")

    # Data for BP chart with adjusted width and attractive colors
    bp_chart = go.Figure()
    bp_chart.add_trace(go.Bar(
        x=['Systolic BP'],
        y=[patient.get('SystolicBP', 0)],  # Default to 0 if not found
        name='Systolic BP',
        marker_color='#69b3a2',
        width=[0.5],
    ))
    bp_chart.add_trace(go.Bar(
        x=['Diastolic BP'],
        y=[patient.get('DiastolicBP', 0)],  # Default to 0 if not found
        name='Diastolic BP',
        marker_color='#404080',
        width=[0.5],
    ))
    bp_chart.update_layout(
        barmode='stack', 
        title=f'Blood Pressure Levels for Patient {patient.get("PatientID")}',
        plot_bgcolor='#f7f7f7',
        paper_bgcolor='#f7f7f7'
    )
    bp_chart_html = pio.to_html(bp_chart, full_html=False)

    # Data for Glucose chart with adjusted width and attractive colors
    glucose_chart = go.Figure()
    glucose_chart.add_trace(go.Bar(
        x=['Fasting Glucose'],
        y=[patient.get('FastingGlucose', 0)],  # Default to 0 if not found
        name='Fasting Glucose',
        marker_color='#ffa07a',
        width=[0.5],
    ))
    glucose_chart.add_trace(go.Bar(
        x=['Postprandial Glucose'],
        y=[patient.get('PostprandialGlucose', 0)],  # Default to 0 if not found
        name='Postprandial Glucose',
        marker_color='#20b2aa',
        width=[0.5],
    ))
    glucose_chart.update_layout(
        barmode='stack', 
        title=f'Glucose Levels for Patient {patient.get("PatientID")}',
        plot_bgcolor='#f7f7f7',
        paper_bgcolor='#f7f7f7'
    )
    glucose_chart_html = pio.to_html(glucose_chart, full_html=False)

    return render_template('dashboard_view.html',
                           patient=patient,
                           bp_chart_html=bp_chart_html,
                           glucose_chart_html=glucose_chart_html)
@app.route("/recommendation", methods=['GET', 'POST'])
def recommendation():
    # Get the patient data from the session
    patient = session.get('patient_data')

    # Ensure the patient data is present in the session
    if not patient:
        return "Patient data not found in session. Please log in again.", 404

    # Extract the PatientID from the session
    patient_id = patient.get('PatientID')

    if request.method == "POST":
        # You can add additional logic if needed for the POST request,
        # such as processing other inputs, etc.

        # Redirect to the Streamlit app with the patient ID as a query parameter
        streamlit_url = f"http://localhost:8501?PatientID={patient_id}"
        return redirect(streamlit_url)
    # For GET requests, render the recommendation page with the patient data if necessary
    return render_template("recommendation.html", patient_id=patient_id,**session['patient_data'])

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route('/register_patient', methods=['POST'])
def register_patient():
    patient_data = {
        'PatientID': request.form.get('PatientID'),
        'Name': request.form.get('Name'),
        'Gender': request.form.get('Gender'),
        'Age': request.form.get('Age'),
        'Address': request.form.get('Address'),
        'ContactDetails': request.form.get('ContactDetails'),
        'Attender': request.form.get('Attender'),
        'SystolicBP': request.form.get('SystolicBP'),
        'DiastolicBP': request.form.get('DiastolicBP'),
        'FastingGlucose': request.form.get('FastingGlucose'),
        'PostprandialGlucose': request.form.get('PostprandialGlucose'),
        'DiagnosedWith': request.form.get('DiagnosedWith'),
        'TypeofBP': request.form.get('TypeofBP'),
        'TypeofDiabetes': request.form.get('TypeofDiabetes'),
        'DoctorReferred': request.form.get('DoctorReferred'),
    }

    # Predict NoofDaysAdmitted using the Random Forest model
    input_features = [
        float(patient_data['Age']),
        float(patient_data['SystolicBP']),
        float(patient_data['DiastolicBP']),
        float(patient_data['FastingGlucose']),
        float(patient_data['PostprandialGlucose'])
    ]
    # Apply the same scaling as during training
    input_features_scaled = scaler.transform([input_features])
    predicted_days = model.predict(input_features_scaled)[0]
    patient_data['NoofDaysAdmitted'] = round(predicted_days)

    # Check if patient ID already exists
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            if any(row['PatientID'] == patient_data['PatientID'] for row in reader):
                return jsonify({'status': 'error', 'message': 'Patient ID already exists'}), 400

    # Save patient data to CSV file
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(patient_data.keys())
        writer.writerow(patient_data.values())

    return jsonify({'status': 'success', 'message': 'Patient registered successfully'})

if __name__ == '__main__':
    app.run(debug=True)
