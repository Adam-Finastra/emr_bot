import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import hashlib
from statsmodels.tsa.seasonal import seasonal_decompose
import json
from collections import defaultdict
import joblib
import os
from typing import Dict, List, Any, Optional

class AIModule:
    def __init__(self):
        # Initialize data preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Learning Agent Components
        self.patient_history = defaultdict(list)
        self.recommendation_history = defaultdict(list)
        self.outcome_history = defaultdict(list)
        self.model_path = 'models/patient_risk_model.joblib'
        self.scaler_path = 'models/scaler.joblib'
        self.load_or_initialize_models()
        
        # Utility-Based Decision Making
        self.utility_weights = {
            'critical_alert': 1.0,
            'high_risk': 0.8,
            'moderate_risk': 0.5,
            'low_risk': 0.2
        }
        
        # Goal-Based Components
        self.clinical_goals = {
            'early_diagnosis': {
                'weight': 0.8,
                'indicators': ['trend_analysis', 'risk_factors', 'symptom_patterns'],
                'thresholds': {
                    'risk_increase': 0.2,
                    'symptom_frequency': 3
                }
            },
            'emergency_detection': {
                'weight': 1.0,
                'indicators': ['vital_signs', 'pain_scale', 'consciousness'],
                'thresholds': {
                    'critical_vitals': 2,
                    'pain_threshold': 7
                }
            },
            'preventive_care': {
                'weight': 0.6,
                'indicators': ['lifestyle_factors', 'family_history', 'age_risk'],
                'thresholds': {
                    'risk_score': 0.4,
                    'age_threshold': 50
                }
            }
        }
        
        # Memory Module with enhanced structure
        self.patient_memory = defaultdict(lambda: {
            'visits': [],
            'recommendations': [],
            'outcomes': [],
            'risk_trends': [],
            'compliance': [],
            'symptom_patterns': defaultdict(int),
            'treatment_effectiveness': defaultdict(float),
            'last_assessment': None
        })
        
        # Initialize clinical ranges
        self._initialize_clinical_ranges()

    def _initialize_clinical_ranges(self):
        """Initialize clinical ranges and thresholds"""
        # BMI ranges
        self.bmi_ranges = {
            'Severely Underweight': {'range': (0, 16), 'risk': 'HIGH'},
            'Underweight': {'range': (16, 18.5), 'risk': 'MODERATE'},
            'Normal': {'range': (18.5, 24.9), 'risk': 'LOW'},
            'Overweight': {'range': (25, 29.9), 'risk': 'MODERATE'},
            'Obesity Class I': {'range': (30, 34.9), 'risk': 'HIGH'},
            'Obesity Class II': {'range': (35, 39.9), 'risk': 'HIGH'},
            'Obesity Class III': {'range': (40, float('inf')), 'risk': 'CRITICAL'}
        }
        
        # Blood Pressure ranges
        self.bp_ranges = {
            'Normal': {'range': (0, 120, 0, 80), 'risk': 'LOW'},
            'Elevated': {'range': (120, 129, 0, 80), 'risk': 'MODERATE'},
            'Hypertension Stage 1': {'range': (130, 139, 80, 89), 'risk': 'HIGH'},
            'Hypertension Stage 2': {'range': (140, float('inf'), 90, float('inf')), 'risk': 'HIGH'},
            'Hypertensive Crisis': {'range': (180, float('inf'), 120, float('inf')), 'risk': 'CRITICAL'},
            'Hypotension': {'range': (0, 90, 0, 60), 'risk': 'HIGH'}
        }
        
        # Temperature ranges
        self.temp_ranges = {
            'Hypothermia': {'range': (0, 95), 'risk': 'CRITICAL'},
            'Low Normal': {'range': (95, 97), 'risk': 'MODERATE'},
            'Normal': {'range': (97, 99), 'risk': 'LOW'},
            'Low Grade Fever': {'range': (99, 100.4), 'risk': 'MODERATE'},
            'Fever': {'range': (100.4, 103), 'risk': 'HIGH'},
            'High Fever': {'range': (103, float('inf')), 'risk': 'CRITICAL'}
        }
        
        # Pulse ranges
        self.pulse_ranges = {
            'Bradycardia': {'range': (0, 60), 'risk': 'HIGH'},
            'Normal': {'range': (60, 100), 'risk': 'LOW'},
            'Tachycardia': {'range': (100, float('inf')), 'risk': 'HIGH'}
        }

    def load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        os.makedirs('models', exist_ok=True)
        
        # Load or initialize ML model
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                # Initialize with dummy data to ensure model is properly initialized
                dummy_X = np.array([[0, 0, 0, 0, 0]])
                dummy_y = np.array([0])
                self.model.fit(dummy_X, dummy_y)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            # Initialize with dummy data
            dummy_X = np.array([[0, 0, 0, 0, 0]])
            dummy_y = np.array([0])
            self.model.fit(dummy_X, dummy_y)
        
        # Load or initialize scaler
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
            except:
                self.scaler = StandardScaler()
                # Initialize with dummy data
                dummy_data = np.array([[0, 0, 0, 0, 0]])
                self.scaler.fit(dummy_data)
        else:
            self.scaler = StandardScaler()
            # Initialize with dummy data
            dummy_data = np.array([[0, 0, 0, 0, 0]])
            self.scaler.fit(dummy_data)

    def preprocess_data(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess patient data for analysis"""
        features = [
            patient_data['systolic_bp'],
            patient_data['diastolic_bp'],
            patient_data['pulse'],
            patient_data['temp'],
            patient_data['height'],
            patient_data['weight'],
            patient_data['age']
        ]
        return self.scaler.transform(np.array(features).reshape(1, -1))

    def update_patient_history(self, registration_id: str, visit_data: Dict[str, Any]):
        """Update patient history with new visit data"""
        self.patient_history[registration_id].append(visit_data)
        self._update_memory_module(registration_id, visit_data)
        self._update_learning_model(registration_id)

    def _update_memory_module(self, registration_id: str, visit_data: Dict[str, Any]):
        """Update the memory module with new visit information"""
        memory = self.patient_memory[registration_id]
        
        # Update visit history
        visit_record = {
            'timestamp': datetime.now().isoformat(),
            'vitals': visit_data,
            'risk_score': self.calculate_risk_score(visit_data)['score'],
            'symptoms': visit_data.get('symptoms', []),
            'treatments': visit_data.get('treatments', [])
        }
        memory['visits'].append(visit_record)
        
        # Update symptom patterns
        for symptom in visit_data.get('symptoms', []):
            memory['symptom_patterns'][symptom] += 1
        
        # Keep only last 10 visits for efficiency
        memory['visits'] = memory['visits'][-10:]
        
        # Update risk trends
        self._update_risk_trends(registration_id)
        
        # Update treatment effectiveness
        self._update_treatment_effectiveness(registration_id, visit_data)

    def _update_risk_trends(self, registration_id: str):
        """Update risk trends based on historical data"""
        memory = self.patient_memory[registration_id]
        if len(memory['visits']) >= 2:
            recent_risks = [visit['risk_score'] for visit in memory['visits']]
            trend = np.polyfit(range(len(recent_risks)), recent_risks, 1)[0]
            memory['risk_trends'].append({
                'timestamp': datetime.now().isoformat(),
                'trend': trend
            })
            memory['risk_trends'] = memory['risk_trends'][-5:]  # Keep last 5 trends

    def _update_treatment_effectiveness(self, registration_id: str, visit_data: Dict[str, Any]):
        """Update treatment effectiveness based on outcomes"""
        memory = self.patient_memory[registration_id]
        if len(memory['visits']) >= 2:
            current_visit = memory['visits'][-1]
            previous_visit = memory['visits'][-2]
            
            for treatment in previous_visit.get('treatments', []):
                if treatment in current_visit.get('treatments', []):
                    effectiveness = self._calculate_treatment_effectiveness(
                        previous_visit, current_visit, treatment
                    )
                    memory['treatment_effectiveness'][treatment] = effectiveness

    def _calculate_treatment_effectiveness(self, previous_visit: Dict, current_visit: Dict, treatment: str) -> float:
        """Calculate effectiveness of a treatment"""
        # Calculate improvement in symptoms
        previous_symptoms = set(previous_visit.get('symptoms', []))
        current_symptoms = set(current_visit.get('symptoms', []))
        
        # Calculate symptom improvement
        resolved_symptoms = previous_symptoms - current_symptoms
        new_symptoms = current_symptoms - previous_symptoms
        
        # Calculate effectiveness score
        effectiveness = len(resolved_symptoms) / (len(previous_symptoms) + 1e-6)
        effectiveness -= len(new_symptoms) * 0.5  # Penalize new symptoms
        
        return max(0.0, min(1.0, effectiveness))

    def calculate_utility_score(self, recommendation: Dict[str, Any], patient_data: Dict[str, Any]) -> float:
        """Calculate utility score for a recommendation based on patient context"""
        base_score = self.utility_weights.get(recommendation['risk_level'], 0.3)
        
        # Adjust based on patient history
        if patient_data['registration_id'] in self.patient_memory:
            memory = self.patient_memory[patient_data['registration_id']]
            
            # Consider risk trends
            if memory['risk_trends'] and memory['risk_trends'][-1]['trend'] > 0:
                base_score *= 1.2
            
            # Consider treatment effectiveness
            if 'treatment' in recommendation:
                effectiveness = memory['treatment_effectiveness'].get(
                    recommendation['treatment'], 0.5
                )
                base_score *= (1 + effectiveness)
        
        # Adjust based on clinical goals
        goal_alignment = self._get_goal_alignment(recommendation)
        for goal in goal_alignment:
            base_score *= self.clinical_goals[goal]['weight']
        
        return round(base_score, 2)

    def _get_goal_alignment(self, recommendation: Dict[str, Any]) -> List[str]:
        """Determine which clinical goals the recommendation aligns with"""
        aligned_goals = []
        for goal, info in self.clinical_goals.items():
            if any(indicator in recommendation for indicator in info['indicators']):
                if self._meets_goal_thresholds(recommendation, goal):
                    aligned_goals.append(goal)
        return aligned_goals

    def _meets_goal_thresholds(self, recommendation: Dict[str, Any], goal: str) -> bool:
        """Check if recommendation meets goal-specific thresholds"""
        thresholds = self.clinical_goals[goal]['thresholds']
        
        if goal == 'early_diagnosis':
            return (
                recommendation.get('risk_increase', 0) >= thresholds['risk_increase'] or
                recommendation.get('symptom_frequency', 0) >= thresholds['symptom_frequency']
            )
        elif goal == 'emergency_detection':
            return (
                recommendation.get('critical_vitals', 0) >= thresholds['critical_vitals'] or
                recommendation.get('pain_scale', 0) >= thresholds['pain_threshold']
            )
        elif goal == 'preventive_care':
            return (
                recommendation.get('risk_score', 0) >= thresholds['risk_score'] or
                recommendation.get('age', 0) >= thresholds['age_threshold']
            )
        return False

    def generate_recommendations(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations with utility scoring and goal alignment"""
        recommendations = []
        risk_assessment = self.calculate_risk_score(patient_data)
        
        # Generate base recommendations
        base_recommendations = self._generate_base_recommendations(patient_data)
        
        # Enhance recommendations with utility scores and goal alignment
        for rec in base_recommendations:
            utility_score = self.calculate_utility_score(rec, patient_data)
            goal_alignment = self._get_goal_alignment(rec)
            
            enhanced_rec = {
                **rec,
                'utility_score': utility_score,
                'goal_alignment': goal_alignment,
                'priority': self._calculate_priority(utility_score, goal_alignment)
            }
            recommendations.append(enhanced_rec)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        # Update recommendation history
        if patient_data.get('registration_id'):
            self.recommendation_history[patient_data['registration_id']].append({
                'timestamp': datetime.now().isoformat(),
                'recommendations': recommendations
            })
        
        return recommendations

    def _calculate_priority(self, utility_score: float, goal_alignment: List[str]) -> float:
        """Calculate overall priority score for a recommendation"""
        priority = utility_score
        
        # Boost priority for emergency-related goals
        if 'emergency_detection' in goal_alignment:
            priority *= 1.5
        
        # Boost priority for early diagnosis
        if 'early_diagnosis' in goal_alignment:
            priority *= 1.2
        
        return round(priority, 2)

    def _generate_base_recommendations(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate base recommendations based on patient data"""
        recommendations = []
        
        # Analyze vital signs
        vital_signs = self._analyze_vital_signs(patient_data)
        recommendations.extend(vital_signs)
        
        # Analyze symptoms
        symptoms = self._analyze_symptoms(patient_data)
        recommendations.extend(symptoms)
        
        # Analyze risk factors
        risk_factors = self._analyze_risk_factors(patient_data)
        recommendations.extend(risk_factors)
        
        return recommendations

    def _analyze_vital_signs(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze vital signs and generate recommendations"""
        recommendations = []
        
        # Blood Pressure Analysis
        bp_category = self.analyze_bp(
            patient_data['systolic_bp'],
            patient_data['diastolic_bp'],
            patient_data['age']
        )
        if bp_category in ['Hypertension Stage 2', 'Hypertensive Crisis']:
            recommendations.append({
                'type': 'vital_signs',
                'category': 'blood_pressure',
                'risk_level': 'HIGH',
                'message': f'Critical Alert: {bp_category} detected',
                'action': 'Immediate medical attention required'
            })
        
        # Temperature Analysis
        temp_category = self.analyze_temp(patient_data['temp'])
        if temp_category in ['High Fever', 'Fever']:
            recommendations.append({
                'type': 'vital_signs',
                'category': 'temperature',
                'risk_level': 'HIGH',
                'message': f'Alert: {temp_category} detected',
                'action': 'Monitor temperature and consider antipyretics'
            })
        
        # Pulse Analysis
        pulse_category = self.analyze_pulse(patient_data['pulse'], patient_data['age'])
        if pulse_category in ['Bradycardia', 'Tachycardia']:
            recommendations.append({
                'type': 'vital_signs',
                'category': 'pulse',
                'risk_level': 'MODERATE',
                'message': f'Alert: {pulse_category} detected',
                'action': 'Evaluate for underlying causes'
            })
        
        return recommendations

    def _analyze_symptoms(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze symptoms and generate recommendations"""
        recommendations = []
        
        # Get patient history
        if patient_data.get('registration_id') in self.patient_memory:
            memory = self.patient_memory[patient_data['registration_id']]
            
            # Analyze symptom patterns
            for symptom, frequency in memory['symptom_patterns'].items():
                if frequency >= 3:  # Symptom appears frequently
                    recommendations.append({
                        'type': 'symptoms',
                        'category': 'pattern',
                        'risk_level': 'MODERATE',
                        'message': f'Frequent symptom: {symptom}',
                        'action': 'Consider detailed evaluation'
                    })
        
        return recommendations

    def _analyze_risk_factors(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze risk factors and generate recommendations"""
        recommendations = []
        
        # Age-based risk factors
        if patient_data['age'] >= 50:
            recommendations.append({
                'type': 'risk_factors',
                'category': 'age',
                'risk_level': 'MODERATE',
                'message': 'Age-related risk factors present',
                'action': 'Consider preventive screening'
            })
        
        # BMI-based risk factors
        bmi = self.calculate_bmi(patient_data['height'], patient_data['weight'])
        bmi_category = self.get_bmi_category(bmi)
        if bmi_category in ['Obesity Class II', 'Obesity Class III']:
            recommendations.append({
                'type': 'risk_factors',
                'category': 'bmi',
                'risk_level': 'HIGH',
                'message': f'High BMI category: {bmi_category}',
                'action': 'Consider weight management program'
            })
        
        return recommendations

    def learn_from_outcome(self, registration_id: str, outcome_data: Dict[str, Any]):
        """Learn from patient outcomes to improve future recommendations"""
        if registration_id in self.patient_memory:
            memory = self.patient_memory[registration_id]
            memory['outcomes'].append({
                'timestamp': datetime.now().isoformat(),
                'outcome': outcome_data
            })
            
            # Update model if enough data is available
            if len(memory['outcomes']) >= 5:
                self._update_learning_model(registration_id)

    def _update_learning_model(self, registration_id: str):
        """Update the ML model with new learning data"""
        memory = self.patient_memory[registration_id]
        if len(memory['visits']) >= 5 and len(memory['outcomes']) >= 5:
            try:
                # Prepare training data
                X = []
                y = []
                for visit, outcome in zip(memory['visits'][-5:], memory['outcomes'][-5:]):
                    X.append(self._prepare_features(visit))
                    y.append(outcome['outcome']['success'])
                
                X = np.array(X)
                y = np.array(y)
                
                # Update model
                self.model.fit(X, y)
                
                # Save updated model
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
            except Exception as e:
                print(f"Error updating learning model: {str(e)}")
                # Initialize with dummy data if update fails
                dummy_X = np.array([[0, 0, 0, 0, 0]])
                dummy_y = np.array([0])
                self.model.fit(dummy_X, dummy_y)

    def _prepare_features(self, visit_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for ML model"""
        try:
            features = [
                visit_data['vitals']['systolic_bp'],
                visit_data['vitals']['diastolic_bp'],
                visit_data['vitals']['pulse'],
                visit_data['vitals']['temp'],
                visit_data['risk_score']
            ]
            return np.array(features)
        except KeyError:
            # If visit_data structure is different, try direct access
            features = [
                visit_data.get('systolic_bp', 0),
                visit_data.get('diastolic_bp', 0),
                visit_data.get('pulse', 0),
                visit_data.get('temp', 0),
                visit_data.get('risk_score', 0)
            ]
            return np.array(features)

    def calculate_bmi(self, height_cm: float, weight_kg: float) -> float:
        """Calculate BMI from height and weight"""
        height_m = height_cm / 100
        return round(weight_kg / (height_m ** 2), 1)

    def get_bmi_category(self, bmi: float) -> str:
        """Get BMI category based on calculated BMI"""
        for category, info in self.bmi_ranges.items():
            if info['range'][0] <= bmi < info['range'][1]:
                return category
        return 'Unknown'

    def analyze_bp(self, systolic: int, diastolic: int, age: int) -> str:
        """Analyze blood pressure readings"""
        # Age-adjusted BP analysis
        if age >= 65:
            if systolic < 120 and diastolic < 70:
                return 'Low Normal for Age'
        
        for category, info in self.bp_ranges.items():
            if info['range'][0] <= systolic < info['range'][1] and info['range'][2] <= diastolic < info['range'][3]:
                return category
        return 'Hypertension Stage 2'

    def analyze_temp(self, temp_f: float) -> str:
        """Analyze body temperature"""
        for category, info in self.temp_ranges.items():
            if info['range'][0] <= temp_f < info['range'][1]:
                return category
        return 'Unknown'

    def analyze_pulse(self, pulse: int, age: int) -> str:
        """Analyze pulse rate with age consideration"""
        # Age-adjusted pulse ranges
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

    def calculate_risk_score(self, patient_data: Dict[str, Any], historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Calculate risk score based on clinical guidelines and historical data"""
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
            if isinstance(trend_risk, str) and "increasing" in trend_risk.lower():
                score += 0.2
                risk_factors.append('Deteriorating Trends')
        
        # Determine risk level
        risk_level = 'LOW'
        if score >= 0.8:
            risk_level = 'CRITICAL'
        elif score >= 0.6:
            risk_level = 'HIGH'
        elif score >= 0.3:
            risk_level = 'MODERATE'
        
        return {
            'score': round(score, 2),
            'level': risk_level,
            'factors': risk_factors
        }

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

    def generate_dashboard_data(self, patient_data: Dict[str, Any], historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate comprehensive dashboard data with risk assessment"""
        risk_assessment = self.calculate_risk_score(patient_data, historical_data)
        disease_prediction = self.predict_disease(patient_data)
        
        return {
            'risk_assessment': risk_assessment,
            'disease_prediction': disease_prediction,
            'vital_signs': {
                'bmi': self.calculate_bmi(patient_data['height'], patient_data['weight']),
                'bp_category': self.analyze_bp(patient_data['systolic_bp'], patient_data['diastolic_bp'], patient_data['age']),
                'temp_category': self.analyze_temp(patient_data['temp']),
                'pulse_category': self.analyze_pulse(patient_data['pulse'], patient_data['age'])
            },
            'recommendations': self.generate_recommendations(patient_data)
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

    def predict_disease(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict potential diseases based on patient data"""
        try:
            # Prepare features for prediction
            features = self._prepare_features(patient_data)
            
            # Ensure model is trained
            if not hasattr(self.model, 'estimators_'):
                # Initialize with dummy data if not trained
                dummy_X = np.array([[0, 0, 0, 0, 0]])
                dummy_y = np.array([0])
                self.model.fit(dummy_X, dummy_y)
            
            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1))
            probability = self.model.predict_proba(features.reshape(1, -1))
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(probability[0][1]),
                'confidence': 'high' if probability[0][1] > 0.8 else 'moderate' if probability[0][1] > 0.5 else 'low'
            }
        except Exception as e:
            print(f"Error in disease prediction: {str(e)}")
            return None

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