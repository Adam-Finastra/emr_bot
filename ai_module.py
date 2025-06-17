import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import hashlib
from statsmodels.tsa.seasonal import seasonal_decompose
import json

class AIModule:
    def __init__(self):
        # Enhanced BMI ranges with more detailed categories and health implications
        self.bmi_ranges = {
            'Severely Underweight': {'range': (0, 16), 'risk': 'HIGH', 'implications': ['Malnutrition', 'Weakened immune system', 'Osteoporosis']},
            'Underweight': {'range': (16, 18.5), 'risk': 'MODERATE', 'implications': ['Nutritional deficiencies', 'Fatigue', 'Hormonal imbalances']},
            'Normal': {'range': (18.5, 24.9), 'risk': 'LOW', 'implications': ['Healthy weight range', 'Optimal health status']},
            'Overweight': {'range': (25, 29.9), 'risk': 'MODERATE', 'implications': ['Increased cardiovascular risk', 'Joint stress']},
            'Obesity Class I': {'range': (30, 34.9), 'risk': 'HIGH', 'implications': ['Type 2 diabetes risk', 'Hypertension', 'Sleep apnea']},
            'Obesity Class II': {'range': (35, 39.9), 'risk': 'HIGH', 'implications': ['Severe cardiovascular risk', 'Metabolic syndrome']},
            'Obesity Class III': {'range': (40, float('inf')), 'risk': 'CRITICAL', 'implications': ['Extreme health risks', 'Mobility issues', 'Multiple comorbidities']}
        }
        
        # Enhanced BP ranges with detailed categories and immediate actions
        self.bp_ranges = {
            'Normal': {'range': (0, 120, 0, 80), 'risk': 'LOW', 'action': 'Continue regular monitoring'},
            'Elevated': {'range': (120, 129, 0, 80), 'risk': 'MODERATE', 'action': 'Lifestyle modifications recommended'},
            'Hypertension Stage 1': {'range': (130, 139, 80, 89), 'risk': 'HIGH', 'action': 'Medical consultation required'},
            'Hypertension Stage 2': {'range': (140, float('inf'), 90, float('inf')), 'risk': 'HIGH', 'action': 'Immediate medical attention'},
            'Hypertensive Crisis': {'range': (180, float('inf'), 120, float('inf')), 'risk': 'CRITICAL', 'action': 'Emergency care required'},
            'Hypotension': {'range': (0, 90, 0, 60), 'risk': 'HIGH', 'action': 'Medical evaluation needed'}
        }
        
        # Enhanced temperature ranges with detailed implications
        self.temp_ranges = {
            'Hypothermia': {'range': (0, 95), 'risk': 'CRITICAL', 'implications': ['Organ failure risk', 'Immediate warming needed']},
            'Low Normal': {'range': (95, 97), 'risk': 'MODERATE', 'implications': ['Monitor for further decrease']},
            'Normal': {'range': (97, 99), 'risk': 'LOW', 'implications': ['Healthy range']},
            'Low Grade Fever': {'range': (99, 100.4), 'risk': 'MODERATE', 'implications': ['Possible infection', 'Monitor closely']},
            'Fever': {'range': (100.4, 103), 'risk': 'HIGH', 'implications': ['Likely infection', 'Medical attention needed']},
            'High Fever': {'range': (103, float('inf')), 'risk': 'CRITICAL', 'implications': ['Severe infection risk', 'Emergency care needed']}
        }
        
        # Enhanced pulse ranges with age-specific considerations
        self.pulse_ranges = {
            'Bradycardia': {'range': (0, 60), 'risk': 'HIGH', 'implications': ['Possible heart condition', 'Medication review needed']},
            'Normal': {'range': (60, 100), 'risk': 'LOW', 'implications': ['Healthy range']},
            'Tachycardia': {'range': (100, float('inf')), 'risk': 'HIGH', 'implications': ['Possible underlying condition', 'Stress evaluation needed']}
        }
        
        # Initialize ML components
        self.model = None
        self.label_encoder = LabelEncoder()
        self.risk_factors = set()
        
        # Risk scoring thresholds
        self.risk_thresholds = {
            'LOW': 0.3,
            'MODERATE': 0.6,
            'HIGH': 0.8,
            'CRITICAL': float('inf')
        }
        
        # Clinical guidelines
        self.clinical_guidelines = {
            'hypertension': {
                'normal': {'systolic': (0, 120), 'diastolic': (0, 80)},
                'elevated': {'systolic': (120, 129), 'diastolic': (0, 80)},
                'stage1': {'systolic': (130, 139), 'diastolic': (80, 89)},
                'stage2': {'systolic': (140, float('inf')), 'diastolic': (90, float('inf'))}
            },
            'pulse': {
                'normal': (60, 100),
                'bradycardia': (0, 60),
                'tachycardia': (100, float('inf'))
            }
        }

    def _encrypt_sensitive_data(self, data):
        """Encrypt sensitive patient data using SHA-256"""
        if isinstance(data, dict):
            return {k: self._encrypt_sensitive_data(v) for k, v in data.items()}
        elif isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        return data

    def calculate_bmi(self, height_cm, weight_kg):
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 1)

    def get_bmi_category(self, bmi):
        for category, info in self.bmi_ranges.items():
            if info['range'][0] <= bmi < info['range'][1]:
                return category
        return 'Unknown'

    def analyze_bp(self, systolic, diastolic, age):
        # Age-adjusted BP analysis
        if age >= 65:
            # Adjusted ranges for elderly
            if systolic < 120 and diastolic < 70:
                return 'Low Normal for Age'
        
        for category, info in self.bp_ranges.items():
            if info['range'][0] <= systolic < info['range'][1] and info['range'][2] <= diastolic < info['range'][3]:
                return category
        return 'Hypertension Stage 2'

    def analyze_temp(self, temp_f):
        for category, info in self.temp_ranges.items():
            if info['range'][0] <= temp_f < info['range'][1]:
                return category
        return 'Unknown'

    def analyze_pulse(self, pulse, age):
        # Age-adjusted pulse analysis
        if age < 1:
            normal_range = (120, 160)
        elif age < 3:
            normal_range = (80, 130)
        elif age < 7:
            normal_range = (70, 120)
        elif age < 12:
            normal_range = (60, 100)
        else:
            normal_range = (60, 100)
            
        if pulse < normal_range[0]:
            return 'Bradycardia'
        elif pulse > normal_range[1]:
            return 'Tachycardia'
        return 'Normal'

    def calculate_risk_score(self, patient_data, historical_data=None):
        """Calculate risk score based on clinical guidelines"""
        risk_factors = []
        score = 0.0
        
        # Blood Pressure Risk
        bp_category = self.analyze_bp(patient_data['systolic_bp'], patient_data['diastolic_bp'], patient_data['age'])
        if bp_category in ['Hypertension Stage 2', 'Hypertensive Crisis']:
            score += 0.4
            risk_factors.append('Hypertension')
        
        # BMI Risk
        bmi = self.calculate_bmi(patient_data['height'], patient_data['weight'])
        bmi_category = self.get_bmi_category(bmi)
        if bmi_category in ['Obesity Class II', 'Obesity Class III']:
            score += 0.3
            risk_factors.append('Obesity')
        
        # Temperature Risk
        temp_category = self.analyze_temp(patient_data['temp'])
        if temp_category in ['High Fever', 'Fever']:
            score += 0.2
            risk_factors.append('Fever')
        
        # Pulse Risk
        pulse_category = self.analyze_pulse(patient_data['pulse'], patient_data['age'])
        if pulse_category in ['Bradycardia', 'Tachycardia']:
            score += 0.1
            risk_factors.append('Abnormal Pulse')
        
        # Historical Trend Risk
        if historical_data:
            trend_risk = self.analyze_trends(historical_data)
            score += trend_risk
            if trend_risk > 0.2:
                risk_factors.append('Deteriorating Trends')
        
        # Determine risk level
        risk_level = 'LOW'
        for level, threshold in self.risk_thresholds.items():
            if score <= threshold:
                risk_level = level
                break
        
        return {
            'score': round(score, 2),
            'level': risk_level,
            'factors': risk_factors
        }

    def analyze_trends(self, historical_data):
        if not historical_data or len(historical_data) < 3:
            return "Not enough data for trend analysis."
        df = pd.DataFrame(historical_data)
        trends = []
        if 'systolic_bp' in df.columns:
            bp_trend = np.polyfit(range(len(df)), df['systolic_bp'], 1)[0]
            if bp_trend > 0.5:
                trends.append("Systolic BP is increasing.")
            elif bp_trend < -0.5:
                trends.append("Systolic BP is decreasing.")
        if 'bmi' in df.columns:
            bmi_trend = np.polyfit(range(len(df)), df['bmi'], 1)[0]
            if bmi_trend > 0.2:
                trends.append("BMI is increasing.")
            elif bmi_trend < -0.2:
                trends.append("BMI is decreasing.")
        if 'temp' in df.columns:
            temp_trend = np.polyfit(range(len(df)), df['temp'], 1)[0]
            if temp_trend > 0.2:
                trends.append("Temperature is increasing.")
            elif temp_trend < -0.2:
                trends.append("Temperature is decreasing.")
        return '\n'.join(trends) if trends else "No significant trends detected."

    def generate_summary(self, patient_data):
        bmi = self.calculate_bmi(patient_data['height'], patient_data['weight'])
        bmi_category = self.get_bmi_category(bmi)
        bp_category = self.analyze_bp(patient_data['systolic_bp'], patient_data['diastolic_bp'], patient_data['age'])
        temp_category = self.analyze_temp(patient_data['temp'])
        pulse_category = self.analyze_pulse(patient_data['pulse'], patient_data['age'])
        summary = [
            f"Patient Summary for {patient_data['name']} (Age: {patient_data['age']}, Gender: {patient_data['gender']}):",
            f"- BMI: {bmi:.1f} ({bmi_category})",
            f"- Blood Pressure: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg ({bp_category})",
            f"- Body Temperature: {patient_data['temp']}°F ({temp_category})",
            f"- Pulse Rate: {patient_data['pulse']} bpm ({pulse_category})"
        ]
        return '\n'.join(summary)

    def generate_alerts(self, patient_data):
        alerts = []
        bmi = self.calculate_bmi(patient_data['height'], patient_data['weight'])
        bmi_category = self.get_bmi_category(bmi)
        bp_category = self.analyze_bp(patient_data['systolic_bp'], patient_data['diastolic_bp'], patient_data['age'])
        temp_category = self.analyze_temp(patient_data['temp'])
        pulse_category = self.analyze_pulse(patient_data['pulse'], patient_data['age'])
        # BMI Alerts
        if bmi_category in ['Severely Underweight', 'Underweight']:
            alerts.append("Critical Alert: Underweight BMI detected – Immediate nutritional intervention required.")
        elif bmi_category in ['Obesity Class II', 'Obesity Class III']:
            alerts.append("Critical Alert: Obese BMI detected – High risk of comorbidities.")
        # BP Alerts
        if bp_category == 'Hypertension Stage 2':
            alerts.append("Critical Alert: Hypertension Stage 2 detected – Immediate medical attention required.")
        elif bp_category == 'Hypertensive Crisis':
            alerts.append("Critical Alert: Hypertensive Crisis detected – Emergency care required.")
        elif bp_category == 'Hypotension':
            alerts.append("Alert: Hypotension detected – Monitor for symptoms.")
        # Temperature Alerts
        if temp_category == 'High Fever':
            alerts.append("Critical Alert: High Fever detected – Possible infection.")
        elif temp_category == 'Fever':
            alerts.append("Alert: Fever detected – Monitor and consider antipyretics.")
        elif temp_category == 'Hypothermia':
            alerts.append("Critical Alert: Hypothermia detected – Immediate warming required.")
        # Pulse Alerts
        if pulse_category == 'Bradycardia':
            alerts.append("Alert: Bradycardia detected – Evaluate for fatigue, dizziness.")
        elif pulse_category == 'Tachycardia':
            alerts.append("Alert: Tachycardia detected – Check for palpitations, underlying causes, and stress.")
        return alerts

    def generate_recommendations(self, patient_data):
        recs = []
        bmi = self.calculate_bmi(patient_data['height'], patient_data['weight'])
        bmi_category = self.get_bmi_category(bmi)
        bp_category = self.analyze_bp(patient_data['systolic_bp'], patient_data['diastolic_bp'], patient_data['age'])
        temp_category = self.analyze_temp(patient_data['temp'])
        pulse_category = self.analyze_pulse(patient_data['pulse'], patient_data['age'])
        # BMI Recommendations
        if bmi_category in ['Severely Underweight', 'Underweight']:
            recs.append("Increase calorie intake and monitor nutrition. Consider referral to dietitian.")
        elif bmi_category in ['Obesity Class II', 'Obesity Class III']:
            recs.append("Comprehensive weight management program. Screen for diabetes and cardiovascular risk.")
        # BP Recommendations
        if bp_category in ['Hypertension Stage 1', 'Hypertension Stage 2']:
            recs.append("Lifestyle modifications recommended. Monitor BP twice daily. Consider medication review.")
        elif bp_category == 'Hypertensive Crisis':
            recs.append("Emergency care required for hypertensive crisis.")
        elif bp_category == 'Hypotension':
            recs.append("Increase fluid intake. Review medications. Monitor for dizziness.")
        # Temperature Recommendations
        if temp_category in ['High Fever', 'Fever']:
            recs.append("Monitor temperature every 4 hours. Consider antipyretics. Screen for infection.")
        elif temp_category == 'Hypothermia':
            recs.append("Gradual warming required. Monitor core temperature. Check for underlying causes.")
        # Pulse Recommendations
        if pulse_category == 'Bradycardia':
            recs.append("Evaluate for fatigue, dizziness.")
        elif pulse_category == 'Tachycardia':
            recs.append("Check for palpitations, underlying causes, and stress.")
        return recs

    def generate_dashboard_data(self, patient_data, historical_data=None):
        """Generate comprehensive dashboard data with risk assessment"""
        risk_assessment = self.calculate_risk_score(patient_data, historical_data)
        disease_prediction = self.predict_disease(patient_data)
        
        # Generate alerts with priority
        alerts = self.generate_alerts(patient_data)
        prioritized_alerts = []
        for alert in alerts:
            if 'Critical Alert' in alert:
                prioritized_alerts.append({'text': alert, 'priority': 'CRITICAL'})
            else:
                prioritized_alerts.append({'text': alert, 'priority': 'ALERT'})
        
        # Sort alerts by priority
        prioritized_alerts.sort(key=lambda x: x['priority'] == 'CRITICAL', reverse=True)
        
        return {
            'patient_info': {
                'registration_id': patient_data['registration_id'],
                'name': patient_data['name'],
                'age': patient_data['age'],
                'gender': patient_data['gender']
            },
            'vitals': {
                'bmi': self.calculate_bmi(patient_data['height'], patient_data['weight']),
                'bp': f"{patient_data['systolic_bp']}/{patient_data['diastolic_bp']}",
                'temp': patient_data['temp'],
                'pulse': patient_data['pulse']
            },
            'risk_assessment': risk_assessment,
            'disease_prediction': disease_prediction,
            'alerts': prioritized_alerts,
            'recommendations': self.generate_recommendations(patient_data),
            'historical_trends': self.analyze_trends(historical_data) if historical_data else None
        }

    def train_ml_model(self, data_path):
        """Enhanced ML model training with time-series features"""
        try:
            df = pd.read_excel(data_path)
            
            # Basic features
            features = ['bmi', 'temp', 'systolic_bp', 'diastolic_bp', 'pulse', 'age']
            
            # Add time-series features if available
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Calculate trends
                df['bp_trend'] = df['systolic_bp'].rolling(window=3).mean()
                df['bmi_trend'] = df['bmi'].rolling(window=3).mean()
                
                features.extend(['bp_trend', 'bmi_trend'])
            
            # Add clinical features
            if 'comorbidities' in df.columns:
                df['comorbidity_count'] = df['comorbidities'].apply(lambda x: len(json.loads(x)) if isinstance(x, str) else 0)
                features.append('comorbidity_count')
            
            if 'medications' in df.columns:
                df['medication_count'] = df['medications'].apply(lambda x: len(json.loads(x)) if isinstance(x, str) else 0)
                features.append('medication_count')
            
            X = df[features].fillna(0)
            y = self.label_encoder.fit_transform(df['disease'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Enhanced model parameters
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None

    def predict_disease(self, patient_data):
        """Enhanced disease prediction with differential diagnosis"""
        if not self.model:
            return "ML model not trained."
            
        try:
            # Prepare features
            features = {
                'bmi': self.calculate_bmi(patient_data['height'], patient_data['weight']),
                'temp': patient_data['temp'],
                'systolic_bp': patient_data['systolic_bp'],
                'diastolic_bp': patient_data['diastolic_bp'],
                'pulse': patient_data['pulse'],
                'age': patient_data['age']
            }
            
            # Add trend features if available
            if 'historical_data' in patient_data:
                historical_df = pd.DataFrame(patient_data['historical_data'])
                features['bp_trend'] = historical_df['systolic_bp'].mean()
                features['bmi_trend'] = historical_df['bmi'].mean()
            
            # Add clinical features if available
            if 'comorbidities' in patient_data:
                features['comorbidity_count'] = len(patient_data['comorbidities'])
            if 'medications' in patient_data:
                features['medication_count'] = len(patient_data['medications'])
            
            # Make prediction
            X = np.array([list(features.values())])
            probabilities = self.model.predict_proba(X)[0]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            predictions = []
            
            for idx in top_3_idx:
                predictions.append({
                    'disease': self.label_encoder.inverse_transform([idx])[0],
                    'confidence': round(probabilities[idx] * 100, 2)
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error in disease prediction: {str(e)}")
            return None