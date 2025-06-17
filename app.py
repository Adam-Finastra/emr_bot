from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import mysql.connector
from config import Config
from ai_module import AIModule
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
ai = AIModule()

# Database connection
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB
        )
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

# Initialize database tables
def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            # Create vital_signs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vital_signs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    registration_id VARCHAR(50) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    gender ENUM('MALE', 'FEMALE') NOT NULL,
                    age INT NOT NULL,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    height FLOAT NOT NULL,
                    weight FLOAT NOT NULL,
                    bmi FLOAT NOT NULL,
                    temp FLOAT NOT NULL,
                    systolic_bp INT NOT NULL,
                    diastolic_bp INT NOT NULL,
                    pulse INT NOT NULL,
                    pain_scale INT NOT NULL,
                    summary TEXT NOT NULL,
                    alerts TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    risk_score FLOAT,
                    risk_level ENUM('LOW', 'MODERATE', 'HIGH', 'CRITICAL'),
                    comorbidities JSON,
                    medications JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create patients table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    registration_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    gender ENUM('MALE', 'FEMALE') NOT NULL,
                    age INT NOT NULL,
                    comorbidities JSON,
                    medications JSON,
                    last_risk_score FLOAT,
                    last_risk_level ENUM('LOW', 'MODERATE', 'HIGH', 'CRITICAL'),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error creating tables: {err}")
        finally:
            cursor.close()
            conn.close()

# Initialize database on startup
init_db()

# Load and train ML model (optional for MVP)
# ai.train_ml_model('vital_signs_disease_dataset_1000.xlsx')

@app.route('/')
def index():
    return render_template('nurse_dashboard.html')

@app.route('/doctor')
def doctor_dashboard():
    try:
        conn = get_db_connection()
        if not conn:
            flash('Database connection error', 'error')
            return render_template('doctor_dashboard.html', patients=[])
        
        cursor = conn.cursor(dictionary=True)
        
        # Get latest vital signs for each patient
        cursor.execute('''
            SELECT v.*, p.name, p.gender, p.age
            FROM vital_signs v
            JOIN patients p ON v.registration_id = p.registration_id
            WHERE v.created_at IN (
                SELECT MAX(created_at)
                FROM vital_signs
                GROUP BY registration_id
            )
            ORDER BY v.created_at DESC
        ''')
        
        patients = cursor.fetchall()
        
        # Process patient data for display
        for patient in patients:
            try:
                patient['alerts'] = json.loads(patient['alerts']) if patient['alerts'] else []
            except Exception:
                patient['alerts'] = []
            # Convert alerts from list of dicts to list of strings if needed
            if patient['alerts'] and isinstance(patient['alerts'][0], dict) and 'text' in patient['alerts'][0]:
                patient['alerts'] = [a['text'] for a in patient['alerts']]
            try:
                patient['recommendations'] = json.loads(patient['recommendations']) if patient['recommendations'] else []
            except Exception:
                patient['recommendations'] = []
            # Ensure summary is always present and correct
            if not patient.get('summary') or not patient['summary'].strip() or patient['summary'].strip().lower() == 'no summary available.':
                # Regenerate summary from latest vitals if missing or placeholder
                try:
                    patient['summary'] = ai.generate_summary({
                        'name': patient.get('name', ''),
                        'age': patient.get('age', 0),
                        'gender': patient.get('gender', ''),
                        'height': patient.get('height', 0),
                        'weight': patient.get('weight', 0),
                        'systolic_bp': patient.get('systolic_bp', 0),
                        'diastolic_bp': patient.get('diastolic_bp', 0),
                        'temp': patient.get('temp', 0),
                        'pulse': patient.get('pulse', 0)
                    })
                except Exception as e:
                    print(f"[DEBUG] Failed to regenerate summary for patient {patient.get('registration_id')}: {e}")
                    patient['summary'] = 'No summary available.'
        
        cursor.close()
        conn.close()
        
        return render_template('doctor_dashboard.html', patients=patients)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('doctor_dashboard.html', patients=[])

@app.route('/submit_vitals', methods=['POST'])
def submit_vitals():
    try:
        # Get and validate JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"Received data: {data}")  # Debug log
        
        # Add current date and time if not provided
        if 'date' not in data:
            data['date'] = datetime.now().strftime('%Y-%m-%d')
        if 'time' not in data:
            data['time'] = datetime.now().strftime('%H:%M:%S')
        
        # Validate required fields
        required_fields = ['registration_id', 'name', 'gender', 'age', 'height', 'weight', 
                         'temp', 'systolic_bp', 'diastolic_bp', 'pulse', 'pain_scale']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Get historical data for trend analysis
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection error'}), 500
        
        cursor = conn.cursor(dictionary=True)
        try:
            # Get historical data
            cursor.execute('''
                SELECT * FROM vital_signs
                WHERE registration_id = %s
                ORDER BY created_at DESC
                LIMIT 10
            ''', (data['registration_id'],))
            historical_data = cursor.fetchall()
            
            # Generate comprehensive analysis
            bmi = ai.calculate_bmi(data['height'], data['weight'])
            summary = ai.generate_summary(data)
            risk_assessment = ai.calculate_risk_score(data, historical_data)
            alerts = ai.generate_alerts(data)
            recommendations = ai.generate_recommendations(data)
            dashboard_data = ai.generate_dashboard_data(data, historical_data)
            print(f"Generated analysis: {dashboard_data}")  # Debug log
            
            # Convert lists to JSON strings for database storage
            alerts_json = json.dumps(alerts)
            recommendations_json = json.dumps(recommendations)
            comorbidities_json = json.dumps(data.get('comorbidities', []))
            medications_json = json.dumps(data.get('medications', []))
            
            # Insert or update patient record
            cursor.execute('''
                INSERT INTO patients (
                    registration_id, name, gender, age,
                    comorbidities, medications, last_risk_score, last_risk_level
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                gender = VALUES(gender),
                age = VALUES(age),
                comorbidities = VALUES(comorbidities),
                medications = VALUES(medications),
                last_risk_score = VALUES(last_risk_score),
                last_risk_level = VALUES(last_risk_level)
            ''', (
                data['registration_id'], data['name'], data['gender'], data['age'],
                comorbidities_json,
                medications_json,
                risk_assessment['score'],
                risk_assessment['level']
            ))
            
            # Insert vital signs with enhanced data
            cursor.execute('''
                INSERT INTO vital_signs (
                    registration_id, name, gender, age, date, time,
                    height, weight, bmi, temp, systolic_bp, diastolic_bp,
                    pulse, pain_scale, summary, alerts, recommendations,
                    risk_score, risk_level, comorbidities, medications
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                data['registration_id'], data['name'], data['gender'], data['age'],
                data['date'], data['time'], data['height'], data['weight'],
                bmi, data['temp'], data['systolic_bp'], data['diastolic_bp'],
                data['pulse'], data['pain_scale'],
                summary,
                alerts_json,
                recommendations_json,
                risk_assessment['score'],
                risk_assessment['level'],
                comorbidities_json,
                medications_json
            ))
            
            conn.commit()
            print("Data successfully saved to database")  # Debug log
            
            # Prepare response data
            response_data = {
                'vitals': {
                    'bmi': bmi,
                    'bp': f"{data['systolic_bp']}/{data['diastolic_bp']}",
                    'temp': data['temp'],
                    'pulse': data['pulse']
                },
                'summary': summary,
                'alerts': alerts,
                'recommendations': recommendations,
                'risk_assessment': risk_assessment
            }
            
            return jsonify(response_data)
        except mysql.connector.Error as err:
            conn.rollback()
            print(f"Database error: {str(err)}")  # Debug log
            return jsonify({'error': f'Database error: {str(err)}'}), 500
        except Exception as e:
            conn.rollback()
            print(f"Unexpected error: {str(e)}")  # Debug log
            return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"Server error: {str(e)}")  # Debug log
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/patient_history/<registration_id>')
def patient_history(registration_id):
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection error'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Get patient info
        cursor.execute('''
            SELECT * FROM patients
            WHERE registration_id = %s
        ''', (registration_id,))
        patient_info = cursor.fetchone()
        
        if not patient_info:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Get vital signs history
        cursor.execute('''
            SELECT * FROM vital_signs
            WHERE registration_id = %s
            ORDER BY created_at DESC
        ''', (registration_id,))
        
        history = cursor.fetchall()
        
        # Process history data
        for record in history:
            try:
                record['alerts'] = json.loads(record['alerts']) if record['alerts'] else []
            except Exception:
                record['alerts'] = []
            try:
                record['recommendations'] = json.loads(record['recommendations']) if record['recommendations'] else []
            except Exception:
                record['recommendations'] = []
            try:
                record['comorbidities'] = json.loads(record['comorbidities']) if record['comorbidities'] else []
            except Exception:
                record['comorbidities'] = []
            try:
                record['medications'] = json.loads(record['medications']) if record['medications'] else []
            except Exception:
                record['medications'] = []
            if not record.get('summary'):
                record['summary'] = ''
        
        # Generate trend analysis
        trend_analysis = ai.analyze_trends(history)
        
        response = {
            'patient_info': patient_info,
            'history': history,
            'trend_analysis': trend_analysis
        }
        
        cursor.close()
        conn.close()
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)