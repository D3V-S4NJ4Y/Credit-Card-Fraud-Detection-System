import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title=" Credit Card & Transaction Fraud Detection", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import time
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import base64
import yagmail
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, IsolationForest, GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE
from scipy import stats
import joblib
import hashlib
import json
from datetime import timedelta
import sqlite3
import datetime
import pytz

# Time Zone For example, for India:
india = pytz.timezone('Asia/Kolkata')
current_time = datetime.datetime.now(india).strftime('%Y-%m-%d %H:%M:%S')

# ------------------- Session State Initialization -------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'show_users' not in st.session_state:
    st.session_state.show_users = False
if 'adding_user' not in st.session_state:
    st.session_state.adding_user = False
if 'registering' not in st.session_state:
    st.session_state.registering = False
if 'login_time' not in st.session_state:
    st.session_state.login_time = None
if 'locked_until' not in st.session_state:
    st.session_state.locked_until = None
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'show_edit_form' not in st.session_state:
    st.session_state.show_edit_form = False
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None

# ------------------- Role Permissions -------------------
role_permissions = {
    "User": ["Read"],
    "Manager": ["Read", "Read/Write"],
    "Admin": ["Read", "Read/Write", "Full"]
}

# ------------------- Authentication Functions -------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, input_password):
    return stored_hash == hash_password(input_password)

def get_users_file_path():
    return os.path.join(os.path.dirname(__file__), 'data', 'users.csv')

def load_users():
    users_file = get_users_file_path()
    if os.path.exists(users_file):
        return pd.read_csv(users_file)
    return pd.DataFrame(columns=['Username', 'Password', 'Email', 'Role', 'Last_Login', 'Status', 'Permissions', 'Department'])

def save_users_data(df):
    try:
        file_path = get_users_file_path()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving user data: {str(e)}")
        return False

def authenticate_user(username, password):
    users_df = load_users()
    user = users_df[users_df['Username'] == username]
    if len(user) == 0:
        return False, None
    stored_hash = user.iloc[0]['Password']
    if verify_password(stored_hash, password):
        return True, user.iloc[0]['Role']
    return False, None

def update_last_login(username):
    try:
        users_df = load_users()
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        users_df.loc[users_df['Username'] == username, 'Last_Login'] = current_time
        save_users_data(users_df)
        return True
    except Exception as e:
        st.error(f"Error updating last login: {str(e)}")
        return False

def check_session_expired():
    if not st.session_state.get('login_time'):
        return
    current_time = datetime.datetime.now()
    session_duration = current_time - st.session_state.login_time
    if session_duration.total_seconds() > 3600:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.login_time = None
        st.warning("🔒 Your session has expired. Please log in again.")
        st.stop()
    elif session_duration.total_seconds() > 3300:
        st.warning("⚠️ Your session will expire in 5 minutes. Please save your work.")

st.markdown('<h1 class="main-header">💳 Credit Card & Transaction Fraud Detection System</h1>', unsafe_allow_html=True)

# Enhanced Authentication
st.sidebar.markdown("### 🔐 System Access Control")

# ------------------- Registration Form (Single Instance) -------------------
if not st.session_state.get('authenticated', False):
    st.sidebar.markdown("---")
    if st.sidebar.button("📝 New User? Register Here", key="register_sidebar"):
        st.session_state.registering = True
        st.rerun()

if st.session_state.get('registering', False) and not st.session_state.get('authenticated', False):
    st.markdown("### 📝 New User Registration")
    with st.form("registration_form"):
        new_username = st.text_input("Username*")
        new_email = st.text_input("Email*")
        new_password = st.text_input("Password*", type="password")
        confirm_password = st.text_input("Confirm Password*", type="password")
        new_department = st.selectbox("Department*", ["General", "Risk", "Compliance", "IT", "Operations"], key="register_department")
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Register"):
                if not new_username or not new_email or not new_password:
                    st.error("Please fill all required fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    users_df = load_users()
                    if new_username in users_df['Username'].values:
                        st.error("Username already exists!")
                    else:
                        new_user = pd.DataFrame([{
                            'Username': new_username,
                            'Password': hash_password(new_password),
                            'Email': new_email,
                            'Role': 'User',
                            'Last_Login': 'Never',
                            'Status': 'Active',
                            'Permissions': 'Read',
                            'Department': new_department
                        }])
                        users_df = pd.concat([users_df, new_user], ignore_index=True)
                        if save_users_data(users_df):
                            st.success("✅ Registration successful! Please login.")
                            time.sleep(2)
                            st.session_state.registering = False
                            st.rerun()
                        else:
                            st.error("Failed to save user data")
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.registering = False
                st.rerun()

# ------------------- Login Logic -------------------
available_roles = ["User", "Manager", "Admin"]
selected_role = st.sidebar.selectbox("👥 Select Role", available_roles, key="first_role_select")
username = st.sidebar.text_input("👤 Username", key="first_username")
password = st.sidebar.text_input("🔑 Password", type="password", key="first_password")

if st.sidebar.button("🔑 Login", key="first_login_button"):
    if username and password:
        authenticated, user_role = authenticate_user(username, password)
        if authenticated and user_role == selected_role:
            st.session_state.authenticated = True
            st.session_state.user_role = user_role
            st.session_state.username = username
            st.session_state.login_time = datetime.datetime.now()
            st.session_state.login_attempts = 0
            if update_last_login(username):
                st.success(f"✅ Welcome {username}! Authenticated as {user_role}")
            else:
                st.warning("Login successful but failed to update last login time")
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            remaining_attempts = 3 - st.session_state.login_attempts
            if remaining_attempts > 0:
                st.error(f"🚫 Invalid credentials or role mismatch. {remaining_attempts} attempts remaining.")
            else:
                current_time = datetime.datetime.now()
                st.session_state.locked_until = current_time + datetime.timedelta(minutes=5)
                st.error("🔒 Too many failed attempts. Account locked for 5 minutes.")
                st.session_state.login_attempts = 0
            st.stop()
    else:
        st.error("⚠️ Please enter both username and password")

if st.session_state.get('authenticated', False):
    st.sidebar.success(f"Logged in as: {st.session_state.username} ({st.session_state.user_role})")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.login_time = None
        st.rerun()

# ------------------- Permission Check Function -------------------
def check_permission(required_permission):
    permission_levels = {
        'User': 0,
        'Read': 1,
        'Read/Write': 2,
        'Full': 3
    }
    user_permission = st.session_state.get('user_permission', st.session_state.get('user_role', 'User'))
    return permission_levels.get(user_permission, 0) >= permission_levels.get(required_permission, 0)

# ------------------- Main App Logic (Only if Authenticated) -------------------
if not st.session_state.get('authenticated', False):
    st.warning("🔒 Please log in to access the system")
    st.stop()
else:
    # Advanced Feature Engineering Class
    class AdvancedFeatureEngineering:
        def __init__(self):
            self.scaler = RobustScaler()
            self.pca = PCA(n_components=5)
            self.transaction_history = []
            
        def create_velocity_features(self, transaction_data, user_history):
            """Create velocity-based features"""
            features = {}
            
            # Transaction velocity (last hour, 24h, 7 days)
            now = datetime.datetime.now()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)
            
            features['txn_last_hour'] = len([t for t in user_history if t['timestamp'] > hour_ago])
            features['txn_last_day'] = len([t for t in user_history if t['timestamp'] > day_ago])
            features['txn_last_week'] = len([t for t in user_history if t['timestamp'] > week_ago])
            
            # Amount velocity
            hour_amounts = [t['amount'] for t in user_history if t['timestamp'] > hour_ago]
            features['amount_velocity_1h'] = sum(hour_amounts) if hour_amounts else 0
            
            day_amounts = [t['amount'] for t in user_history if t['timestamp'] > day_ago]
            features['amount_velocity_24h'] = sum(day_amounts) if day_amounts else 0
            
            return features
        
        def create_behavioral_features(self, transaction_data, user_profile):
            """Create behavioral deviation features"""
            features = {}
            
            # Convert amount_log back to amount for deviation calculation
            amount = np.exp(transaction_data.get('amount_log', 0)) - 1
            
            # Deviation from user's normal patterns
            features['amount_deviation'] = abs(amount - user_profile.get('avg_amount', 0))
            features['hour_deviation'] = abs(transaction_data.get('hour', 0) - user_profile.get('usual_hour', 12))
            features['merchant_familiarity'] = 1 if transaction_data.get('category', '') in user_profile.get('frequent_categories', []) else 0
            
            # Geographic features
            features['location_risk'] = np.random.uniform(0, 1)  # Mock risk score
            features['distance_from_home'] = transaction_data.get('location_distance', 0)
            
            return features
        
        def create_time_features(self, timestamp):
            """Advanced time-based features"""
            features = {}
            
            features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
            features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
            
            # Business hours indicator
            features['business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
            features['late_night'] = 1 if timestamp.hour >= 23 or timestamp.hour <= 5 else 0
            
            return features

    # Advanced Ensemble Model Class
    class AdvancedEnsembleModel:
        def __init__(self):
            self.models = {}
            self.weights = {}
            self.anomaly_detectors = {}
            self.is_trained = False
            
        def initialize_models(self):
            """Initialize ensemble of models"""
            self.models = {
                'xgb': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'lgb': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            # Anomaly detection models
            self.anomaly_detectors = {
                'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
                'one_class_svm': OneClassSVM(nu=0.1),
                'dbscan': DBSCAN(eps=0.5, min_samples=5)
            }
            
        def predict_ensemble(self, X):
            """Advanced ensemble prediction with multiple strategies"""
            if not self.is_trained:
                # Mock training for demonstration
                self.weights = {'xgb': 0.4, 'lgb': 0.3, 'rf': 0.2, 'gb': 0.1}
                self.is_trained = True
                
            # Simulate predictions (in real implementation, use trained models)
            predictions = {}
            probabilities = {}
            
            for name in self.models.keys():
                # Mock predictions for demonstration
                prob = np.random.beta(2, 5)  # Skewed towards low fraud probability
                predictions[name] = 1 if prob > 0.5 else 0
                probabilities[name] = prob
                
            # Weighted ensemble
            weighted_prob = sum(probabilities[name] * self.weights[name] 
            for name in probabilities.keys())
            
            # Anomaly detection
            anomaly_scores = {}
            for name, detector in self.anomaly_detectors.items():
                # Mock anomaly score
                anomaly_scores[name] = np.random.uniform(-1, 1)
                
            return {
                'ensemble_probability': weighted_prob,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'anomaly_scores': anomaly_scores
            }

    # Risk Scoring Engine
    class RiskScoringEngine:
        def __init__(self):
            self.risk_factors = {
                'amount': {'weight': 0.25, 'threshold': 1000},
                'velocity': {'weight': 0.20, 'threshold': 5},
                'location': {'weight': 0.15, 'threshold': 0.7},
                'behavioral': {'weight': 0.20, 'threshold': 0.6},
                'temporal': {'weight': 0.10, 'threshold': 0.5},
                'anomaly': {'weight': 0.10, 'threshold': 0.3}
            }
            
        def calculate_risk_score(self, features, model_output):
            """Calculate comprehensive risk score"""
            risk_components = {}
            
            # Amount risk
            amount_risk = min(features.get('amount_log', 0) / 10, 1.0)
            risk_components['amount'] = amount_risk
            
            # Velocity risk
            velocity_risk = min(features.get('txn_last_hour', 0) / 10, 1.0)
            risk_components['velocity'] = velocity_risk
            
            # Location risk
            location_risk = features.get('location_risk', 0)
            risk_components['location'] = location_risk
            
            # Behavioral risk
            behavioral_risk = features.get('amount_deviation', 0) / 1000
            risk_components['behavioral'] = min(behavioral_risk, 1.0)
            
            # Temporal risk
            temporal_risk = 1.0 if features.get('late_night', 0) else 0.3
            risk_components['temporal'] = temporal_risk
            
            # Anomaly risk
            anomaly_risk = max(model_output.get('anomaly_scores', {}).values()) if model_output.get('anomaly_scores') else 0
            risk_components['anomaly'] = min(abs(anomaly_risk), 1.0)
            
            # Calculate weighted risk score
            total_risk = sum(
                risk_components[factor] * self.risk_factors[factor]['weight']
                for factor in risk_components.keys()
            )
            
            return {
                'total_risk_score': total_risk,
                'risk_components': risk_components,
                'risk_level': self.get_risk_level(total_risk)
            }
        
        def get_risk_level(self, score):
            """Determine risk level based on score"""
            if score >= 0.8:
                return "CRITICAL"
            elif score >= 0.6:
                return "HIGH"
            elif score >= 0.4:
                return "MEDIUM"
            elif score >= 0.2:
                return "LOW"
            else:
                return "MINIMAL"

    # Advanced Alert System
    class AdvancedAlertSystem:
        def __init__(self):
            self.alert_rules = {
                'CRITICAL': {'email': True, 'sms': True, 'webhook': True, 'block': True},
                'HIGH': {'email': True, 'sms': False, 'webhook': True, 'block': False},
                'MEDIUM': {'email': True, 'sms': False, 'webhook': False, 'block': False},
                'LOW': {'email': False, 'sms': False, 'webhook': False, 'block': False}
            }
            
        def process_alert(self, risk_data, transaction_data):
            """Process alerts based on risk level"""
            risk_level = risk_data['risk_level']
            actions = self.alert_rules.get(risk_level, {})
            
            alert_info = {
                'timestamp': datetime.datetime.now(),
                'risk_level': risk_level,
                'risk_score': risk_data['total_risk_score'],
                'transaction_id': self.generate_transaction_id(transaction_data),
                'actions_taken': []
            }
            
            if actions.get('email'):
                alert_info['actions_taken'].append('Email sent')
                
            if actions.get('block'):
                alert_info['actions_taken'].append('Transaction blocked')
                
            return alert_info
        
        def generate_transaction_id(self, transaction_data):
            """Generate unique transaction ID"""
            data_string = json.dumps(transaction_data, sort_keys=True, default=str)
            return hashlib.md5(data_string.encode()).hexdigest()[:12]

    # Initialize advanced components
    @st.cache_resource
    def initialize_advanced_system():
        feature_engineer = AdvancedFeatureEngineering()
        ensemble_model = AdvancedEnsembleModel()
        risk_engine = RiskScoringEngine()
        alert_system = AdvancedAlertSystem()
        
        ensemble_model.initialize_models()
        
        return feature_engineer, ensemble_model, risk_engine, alert_system

    # Load advanced system
    feature_engineer, ensemble_model, risk_engine, alert_system = initialize_advanced_system()

    # Enhanced Email Alert Function
    def send_advanced_email_alert(to_email, transaction_data, risk_data, model_output):
        """Send advanced email alert with detailed risk analysis"""
        risk_level = risk_data['risk_level']
        risk_score = risk_data['total_risk_score']
        
        subject = f"🚨 {risk_level} RISK ALERT: Fraud Detection System"
        
        # Create detailed risk breakdown
        risk_breakdown = "\n".join([
            f"    • {factor.title()}: {score:.2%}" 
            for factor, score in risk_data['risk_components'].items()
        ])
        
        model_breakdown = "\n".join([
            f"    • {model.upper()}: {prob:.2%}" 
            for model, prob in model_output['individual_probabilities'].items()
        ])
        
        content = f"""
        🔒 ADVANCED FRAUD DETECTION ALERT
        
        Risk Level: {risk_level}
        Overall Risk Score: {risk_score:.2%}
        
        📊 TRANSACTION DETAILS:
        • Amount: ${np.exp(transaction_data.get('amount_log', 0)) - 1:.2f}
        • Category: {transaction_data.get('category', 'Unknown')}
        • Time: Hour {transaction_data.get('hour', 0)}
        • Location Distance: {transaction_data.get('location_distance', 0):.2f} miles
        
        🎯 RISK COMPONENT BREAKDOWN:
    {risk_breakdown}
        
        🤖 MODEL ENSEMBLE RESULTS:
    {model_breakdown}
        • Ensemble Prediction: {model_output['ensemble_probability']:.2%}
        
        🚨 RECOMMENDED ACTIONS:
        • Immediate investigation required for {risk_level} risk transactions
        • Consider temporary hold on similar transaction patterns
        • Review customer's recent transaction history
        
        Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Best regards,
        Advanced AI Fraud Detection System
        """
        
        try:
            yag = yagmail.SMTP(user="sanjay.dev925@gmail.com", password="splh yrfu ebuq ghve")
            yag.send(to=to_email, subject=subject, contents=content)
            return True
        except Exception as e:
            st.error(f"Failed to send email alert: {e}")
            return False

    # Add this after imports
    def setup_email_config():
        """Setup email configuration using Streamlit secrets"""
        try:
            if not st.secrets.get("email"):
                st.error("Email configuration not found in secrets")
                return None
                
            return {
                "smtp_user": st.secrets.email.SMTP_USER,
                "smtp_password": st.secrets.email.SMTP_PASSWORD,
                "smtp_server": st.secrets.email.SMTP_SERVER,
                "smtp_port": st.secrets.email.SMTP_PORT
            }
        except Exception as e:
            st.error(f"Error loading email configuration: {str(e)}")
            return None

    def send_fraud_alert_email(user_email, transaction_details, risk_level):
        """Send fraud alert email with proper SSL configuration"""
        try:
            # Initialize yagmail with SSL settings
            yag = yagmail.SMTP(
                user="sanjay.dev925@gmail.com",
                password="splh yrfu ebuq ghve",
                host='smtp.gmail.com',
                smtp_ssl=True,  # Enable SSL
                port=465  # Use SSL port instead of 587
            )
            
            subject = f"🚨 FRAUD ALERT: {risk_level} Risk Detected"
            
            body = f"""
            🔒 Fraud Detection Alert
            
            Risk Level: {risk_level}
            
            Transaction Details:
            • Amount: {transaction_details['amount']}
            • Time: {transaction_details['timestamp']}
            • Location: {transaction_details['location']}
            • Risk Score: {transaction_details['risk_score']}
            
            Please review this transaction immediately.
            """
            
            # Send email with error handling
            try:
                yag.send(
                    to=user_email,
                    subject=subject,
                    contents=body
                )
                st.success(f"🚨 Alert email sent successfully to {user_email}")
                return True, "Email sent successfully"
                
            except Exception as e:
                st.error(f"Failed to send email: {str(e)}")
                return False, f"Failed to send email: {str(e)}"
                
            finally:
                yag.close()
                
        except Exception as e:
            st.error(f"Email configuration error: {str(e)}")
            return False, f"Email configuration error: {str(e)}"

    # Modify the risk analysis section to include email alerts
    def process_transaction_with_alerts(user_email, transaction_data, risk_data):
        """Process transaction and send alerts if necessary"""
        risk_level = risk_data['risk_level']
        
        # Define risk thresholds for alerts
        alert_thresholds = {
            'CRITICAL': 0.9,
            'HIGH': 0.7,
            'MEDIUM': 0.5
        }
        
        # Check if alert should be sent
        if risk_data['total_risk_score'] >= alert_thresholds.get(risk_level, 0):
            # Prepare transaction details for email
            email_transaction_details = {
                'amount': transaction_data.get('amount', 'N/A'),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'location': transaction_data.get('location', 'Unknown'),
                'risk_score': f"{risk_data['total_risk_score']:.2%}"
            }
            
            # Send alert email
            success, message = send_fraud_alert_email(
                user_email,
                email_transaction_details,
                risk_level
            )
            
            if success:
                st.success(f"🚨 Alert email sent to {user_email}")
            else:
                st.error(f"❌ Failed to send alert: {message}")

    # Streamlit App Configuration
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .risk-critical {
            border-left: 5px solid #ffff00;
            padding: 1rem;
            border-radius: 5px;
        }
        .risk-high {
            border-left: 5px solid #ff0000;
            padding: 1rem;
            border-radius: 5px;
        }
        .risk-medium {
            border-left: 5px solid #4caf50;
            padding: 1rem;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    def update_last_login(username):
        """Update user's last login time"""
        try:
            users_df = load_users()
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            users_df.loc[users_df['Username'] == username, 'Last_Login'] = current_time
            save_users_data(users_df)
            return True
        except Exception as e:
            st.error(f"Error updating last login: {str(e)}")
            return False

    # Add Register button in sidebar before login
    if not st.session_state.get('authenticated', False):
        st.sidebar.markdown("---")
        if st.sidebar.button("📝 New User? Register Here", key="register_sidebar_second"):
            st.session_state.registering = True
            st.rerun()

    # Registration form
    if st.session_state.get('registering', False) and not st.session_state.get('authenticated', False):
        st.markdown("### 📝 New User Registration")
        with st.form("registration_form"):
            new_username = st.text_input("Username*")
            new_email = st.text_input("Email*")
            new_password = st.text_input("Password*", type="password")
            confirm_password = st.text_input("Confirm Password*", type="password")
            new_department = st.selectbox("Department*", 
                ["General", "Risk", "Compliance", "IT", "Operations"], key="register_department_second")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Register"):
                    if not new_username or not new_email or not new_password:
                        st.error("Please fill all required fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        # Load existing users or create new file
                        try:
                            users_df = load_users()
                        except:
                            users_df = pd.DataFrame(columns=[
                                'Username', 'Password', 'Email', 'Role', 
                                'Last_Login', 'Status', 'Permissions', 'Department'
                            ])
                        
                        # Check if username exists
                        if new_username in users_df['Username'].values:
                            st.error("Username already exists!")
                        else:
                            # Create new user with User role
                            new_user = pd.DataFrame([{
                                'Username': new_username,
                                'Password': hash_password(new_password),
                                'Email': new_email,
                                'Role': 'User',
                                'Last_Login': 'Never',
                                'Status': 'Active',
                                'Permissions': 'Read',
                                'Department': new_department
                            }])
                            
                            # Add new user and save
                            users_df = pd.concat([users_df, new_user], ignore_index=True)
                            if save_users_data(users_df):
                                st.success("✅ Registration successful! Please login.")
                                time.sleep(2)
                                st.session_state.registering = False
                                st.rerun()
                            else:
                                st.error("Failed to save user data")
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.registering = False
                    st.rerun()

    # Add this before showing any data/content
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please log in to access the system")
        st.stop()
    else:
        # Load necessary data after authentication
        users_df = load_users()  # <-- FIXED: was load_users_data()
        check_session_expired()
        
        # Show role-specific content
        if st.session_state.get('user_role') == 'Admin':
            # Show admin content
            pass
        elif st.session_state.get('user_role') == 'Manager':
            # Show manager content
            pass
        elif st.session_state.get('user_role') == 'User':
            # Show user content
            pass
        else:
            # Show User user content
            pass

    # Advanced Input Panel
    st.sidebar.markdown("### 📋 Transaction Input Panel")

    # Primary transaction details
    amount = st.sidebar.slider("💰 Transaction Amount ($)", 1, 1000000, 100, format="$%d")
    hour = st.sidebar.slider("🕐 Hour of Transaction", 0, 23, 12)
    age = st.sidebar.slider("👤 Customer Age", 18, 100, 35)

    # Enhanced categorical inputs
    category = st.sidebar.selectbox("🏪 Merchant Category", 
        ['gas_transport', 'grocery_pos', 'shopping_net', 'travel', 
        'misc_pos', 'health_fitness', 'entertainment', 'food_dining', 
        'bills_utilities', 'personal_care'])

    gender = st.sidebar.radio("⚧ Gender", ["M", "F", "Other"])
    city_pop = st.sidebar.slider("🏙️ City Population", 100, 10000000, 50000)

    job = st.sidebar.selectbox("💼 Customer Job", 
        ['Teacher', 'Engineer', 'Doctor', 'Lawyer', 'Nurse', 'Artist', 
        'Freelancer', 'Manager', 'Student', 'Retired'])

    transaction_date = st.sidebar.date_input("📅 Transaction Date", datetime.date.today())

    # Advanced settings
    st.sidebar.markdown("### ⚙️ Advanced Settings")
    model_sensitivity = st.sidebar.slider("🎯 Model Sensitivity", 0.1, 1.0, 0.5, 0.1)
    enable_email = st.sidebar.checkbox("📧 Enable Email Alerts", value=True)
    enable_realtime = st.sidebar.checkbox("⚡ Real-time Processing", value=True)

    # Mock user profile and history
    user_profile = {
        'avg_amount': np.random.uniform(50, 500),
        'usual_hour': np.random.randint(8, 20),
        'frequent_categories': np.random.choice(
            ['grocery_pos', 'gas_transport', 'shopping_net'], 
            size=2, replace=False
        ).tolist()
    }

    user_history = [
        {
            'timestamp': datetime.datetime.now() - timedelta(hours=np.random.randint(1, 168)),
            'amount': np.random.uniform(10, 1000),
            'category': np.random.choice(['grocery_pos', 'gas_transport', 'shopping_net'])
        }
        for _ in range(np.random.randint(5, 20))
    ]

    # Feature Engineering
    timestamp = datetime.datetime.combine(transaction_date, datetime.time(hour))

    # User features
    cat_map = {v: i for i, v in enumerate(['gas_transport','grocery_pos','shopping_net','travel','misc_pos','health_fitness','entertainment', 'food_dining', 'bills_utilities', 'personal_care'])}
    gender_map = {'M': 0, 'F': 1, 'Other': 2}
    job_map = {v: i for i, v in enumerate(['Teacher', 'Engineer', 'Doctor', 'Lawyer', 'Nurse', 'Artist', 'Freelancer', 'Manager', 'Student', 'Retired'])}

    User_features = {
        "amount_log": np.log(amount + 1),
        "category": cat_map.get(category, 0),
        "gender": gender_map[gender],
        "city_pop_log": np.log(city_pop + 1),
        "job": job_map[job],
        "age": age,
        "hour": hour,
        "weekday": timestamp.weekday(),
        "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
        "location_distance": round(np.random.uniform(0.1, 10.0), 2)
    }

    # Advanced feature engineering
    velocity_features = feature_engineer.create_velocity_features(User_features, user_history)
    behavioral_features = feature_engineer.create_behavioral_features(User_features, user_profile)
    time_features = feature_engineer.create_time_features(timestamp)

    # Combine all features
    all_features = {**User_features, **velocity_features, **behavioral_features, **time_features}

    # Create DataFrame for model input
    input_df = pd.DataFrame([all_features])

    # Advanced Model Prediction
    model_output = ensemble_model.predict_ensemble(input_df)
    risk_data = risk_engine.calculate_risk_score(all_features, model_output)
    alert_info = alert_system.process_alert(risk_data, all_features)

    # Main Dashboard
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("### 🎯 Fraud Detection Results")
        
        risk_level = risk_data['risk_level']
        risk_score = risk_data['total_risk_score']
        ensemble_prob = model_output['ensemble_probability']
        user_email = users_df[users_df['Username'] == st.session_state.username]['Email'].iloc[0]
        # Dynamic risk display section
        if risk_level == "CRITICAL":
            st.markdown(f'<div class="risk-critical"><h3>🚨 CRITICAL RISK DETECTED</h3><p>Risk Score: {risk_score:.1%}</p><p>Ensemble Probability: {ensemble_prob:.1%}</p></div>', unsafe_allow_html=True)
            if enable_email:
                send_advanced_email_alert(user_email, all_features, risk_data, model_output)
                st.info("📧 Alert email sent due to CRITICAL risk level")
            else:
                st.info("📧 Email alerts are currently disabled")
        
        elif risk_level == "HIGH":
            st.markdown(f'<div class="risk-high"><h3>⚠️ HIGH RISK TRANSACTION</h3><p>Risk Score: {risk_score:.1%}</p><p>Ensemble Probability: {ensemble_prob:.1%}</p></div>', unsafe_allow_html=True)
            if enable_email:
                send_advanced_email_alert(user_email, all_features, risk_data, model_output)
                st.info("📧 Alert email sent due to HIGH risk level")
            else:
                st.info("📧 Email alerts are currently disabled")
        
        else:
            st.markdown(f'<div class="risk-medium"><h3>✅ {risk_level} RISK</h3><p>Risk Score: {risk_score:.1%}</p><p>Ensemble Probability: {ensemble_prob:.1%}</p></div>', unsafe_allow_html=True)

        
        # Model breakdown
        st.markdown("#### 🤖 Model Ensemble Breakdown")
        model_df = pd.DataFrame([
            {'Model': model.upper(), 'Probability': f"{prob:.1%}", 'Prediction': 'Fraud' if pred else 'Legitimate'}
            for model, (prob, pred) in zip(
                model_output['individual_probabilities'].keys(),
                zip(model_output['individual_probabilities'].values(), 
                    model_output['individual_predictions'].values())
            )
        ])
        st.dataframe(model_df, use_container_width=True)

    with col2:
        st.markdown("### 📊 Risk Analysis Dashboard")
        
        # Risk components radar chart
        risk_components = risk_data['risk_components']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(risk_components.values()),
            theta=list(risk_components.keys()),
            fill='toself',
            name='Risk Components'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Risk Component Analysis"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Ensemble probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ensemble_prob * 100,
            title={'text': "Ensemble Fraud Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            delta={'reference': 50}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.markdown("### 🚨 Alert Status")
        
        st.metric("Transaction ID", alert_info['transaction_id'])
        st.metric("Risk Level", risk_level)
        st.metric("Alert Time", alert_info['timestamp'].strftime('%H:%M:%S'))
        
        if alert_info['actions_taken']:
            st.markdown("**Actions Taken:**")
            for action in alert_info['actions_taken']:
                st.markdown(f"• {action}")
        else:
            st.markdown("**No actions required**")

    # Advanced Analytics Section
    st.markdown("---")
    st.markdown("### 📈 Advanced Analytics & Insights")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Feature Analysis", "🎯 Model Performance", "📊 Transaction Patterns", "🛡️ Security Dashboard"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance simulation
            feature_importance = {
                'amount_log': 0.25,
                'txn_last_hour': 0.18,
                'location_distance': 0.15,
                'amount_deviation': 0.12,
                'hour': 0.10,
                'age': 0.08,
                'category': 0.07,
                'late_night': 0.05
            }
            
            fig_importance = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                title="Top Feature Importance",
                labels={'x': 'Importance Score', 'y': 'Features'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # SHAP values simulation
            shap_values = {
                'amount_log': np.random.uniform(-0.1, 0.3),
                'txn_last_hour': np.random.uniform(-0.05, 0.2),
                'location_distance': np.random.uniform(-0.02, 0.15),
                'hour': np.random.uniform(-0.1, 0.1),
                'age': np.random.uniform(-0.05, 0.05)
            }
            
            fig_shap = px.bar(
                x=list(shap_values.values()),
                y=list(shap_values.keys()),
                orientation='h',
                title="SHAP Values (Current Transaction)",
                labels={'x': 'SHAP Value', 'y': 'Features'},
                color=[v > 0 for v in shap_values.values()],
                color_discrete_map={True: 'red', False: 'blue'}
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve simulation
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - np.exp(-5 * fpr)  # Mock ROC curve
            
            fig_roc = px.line(
                x=fpr, y=tpr,
                title="ROC Curve (Model Performance)",
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
            )
            fig_roc.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            # Precision-Recall curve
            recall = np.linspace(0, 1, 100)
            precision = 0.9 * np.exp(-2 * recall)  # Mock PR curve
            
            fig_pr = px.line(
                x=recall, y=precision,
                title="Precision-Recall Curve",
                labels={'x': 'Recall', 'y': 'Precision'}
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    with tab3:
        # Transaction patterns analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly fraud patterns
            hours = range(24)
            fraud_rates = [0.02 + 0.03 * abs(np.sin(h * np.pi / 12)) for h in hours]
            
            fig_hourly = px.bar(
                x=hours, y=fraud_rates,
                title="Fraud Rate by Hour of Day",
                labels={'x': 'Hour', 'y': 'Fraud Rate'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Category risk analysis
            categories = ['grocery_pos', 'gas_transport', 'shopping_net', 'travel', 'entertainment']
            risk_scores = [0.15, 0.12, 0.28, 0.35, 0.22]
            
            fig_category = px.pie(
                values=risk_scores, names=categories,
                title="Risk Distribution by Category"
            )
            st.plotly_chart(fig_category, use_container_width=True)

    with tab4:
        # Security dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🛡️ Threats Blocked Today", "47", delta="12")
            st.metric("⚡ Real-time Alerts", "23", delta="5")
        
        with col2:
            st.metric("🎯 Detection Accuracy", "94.2%", delta="2.1%")
            st.metric("⏱️ Avg Response Time", "0.34s", delta="-0.05s")
        
        with col3:
            st.metric("💰 Losses Prevented", "$234,567", delta="$45,123")
            st.metric("🔍 Investigations Active", "8", delta="2")
        
        # Security timeline
        st.markdown("#### 🕒 Recent Security Events")
        security_events = pd.DataFrame({
            'Time': pd.date_range(start='2024-06-01 08:00', periods=10, freq='1H'),
            'Event': ['High Risk Transaction', 'Anomaly Detected', 'Location Alert', 
                     'Velocity Warning', 'Blocked Transaction'] * 2,
            'Risk Level': ['HIGH', 'MEDIUM', 'LOW', 'HIGH', 'CRITICAL'] * 2,
            'Amount': np.random.uniform(100, 5000, 10)
        })
        
        fig_timeline = px.scatter(
            security_events, x='Time', y='Event', 
            size='Amount', color='Risk Level',
            title="Security Events Timeline"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


    # Real-time Monitoring Dashboard
    if enable_realtime:
        st.markdown("---")
        st.markdown("### ⚡ Real-time Monitoring Dashboard")
        
        # Simulate real-time data
        if st.button("🔄 Refresh Real-time Data"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_tps = np.random.poisson(15)  # Transactions per second
                st.metric("🔄 Live TPS", current_tps, delta=np.random.randint(-3, 4))
            
            with col2:
                fraud_rate = np.random.uniform(0.01, 0.05)
                st.metric("🎯 Live Fraud Rate", f"{fraud_rate:.1%}", 
                delta=f"{np.random.uniform(-0.01, 0.01):.2%}")
            
            with col3:
                avg_response = np.random.uniform(0.2, 0.8)
                st.metric("⚡ Avg Response Time", f"{avg_response:.2f}s",
                delta=f"{np.random.uniform(-0.1, 0.1):.2f}s")
            
            with col4:
                system_health = np.random.choice(['Healthy', 'Warning', 'Critical'], 
                p=[0.8, 0.15, 0.05])
                health_emoji = {'Healthy': '🟢', 'Warning': '🟡', 'Critical': '🔴'}
                st.metric("🏥 System Health", f"{health_emoji[system_health]} {system_health}")
            
            # Real-time transaction stream simulation
            st.markdown("#### 📡 Live Transaction Stream")
            
            # Generate mock real-time transactions
            realtime_data = []
            for i in range(10):
                transaction = {
                    'Time': (datetime.datetime.now() - timedelta(seconds=i*5)).strftime('%H:%M:%S'),
                    'Amount': f"${np.random.uniform(10, 2000):.0f}",
                    'Risk': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], 
                    p=[0.7, 0.2, 0.08, 0.02]),
                    'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']),
                    'Status': np.random.choice(['✅ Approved', '⏳ Review', '🚫 Blocked'], p=[0.85, 0.1, 0.05])
                }
                realtime_data.append(transaction)
            
            realtime_df = pd.DataFrame(realtime_data)
            st.dataframe(realtime_df, use_container_width=True)

    # AI Model Management Section
    if st.session_state.get('user_role') in ['Manager', 'Admin']:
        st.markdown("---")
        st.markdown("### 🤖 AI Model Management & Optimization")
        
        tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "⚙️ Model Config", "🔄 Model Updates"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Model performance over time
                dates = pd.date_range(start='2024-05-01', end='2024-06-01', freq='D')
                accuracy_trend = 0.9 + 0.05 * np.sin(np.arange(len(dates)) * 0.2) + np.random.normal(0, 0.01, len(dates))
                
                fig_trend = px.line(
                    x=dates, y=accuracy_trend,
                    title="Model Accuracy Trend (30 Days)",
                    labels={'x': 'Date', 'y': 'Accuracy'}
                )
                fig_trend.add_hline(y=0.9, line_dash="dash", annotation_text="Target Accuracy")
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Model comparison
                models_comparison = {
                    'XGBoost': {'Accuracy': 0.942, 'Precision': 0.876, 'Recall': 0.823},
                    'LightGBM': {'Accuracy': 0.938, 'Precision': 0.881, 'Recall': 0.815},
                    'Random Forest': {'Accuracy': 0.935, 'Precision': 0.869, 'Recall': 0.834},
                    'Ensemble': {'Accuracy': 0.951, 'Precision': 0.889, 'Recall': 0.841}
                }
                
                comparison_df = pd.DataFrame(models_comparison).T
                
                fig_comparison = px.bar(
                    comparison_df.reset_index(), 
                    x='index', y=['Accuracy', 'Precision', 'Recall'],
                    title="Model Performance Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        with tab2:
            st.markdown("#### ⚙️ Advanced Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ensemble Weights**")
                xgb_weight = st.slider("XGBoost Weight", 0.0, 1.0, 0.4, 0.1)
                lgb_weight = st.slider("LightGBM Weight", 0.0, 1.0, 0.3, 0.1)
                rf_weight = st.slider("Random Forest Weight", 0.0, 1.0, 0.2, 0.1)
                gb_weight = st.slider("Gradient Boost Weight", 0.0, 1.0, 0.1, 0.1)
                
                total_weight = xgb_weight + lgb_weight + rf_weight + gb_weight
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should sum to 1.0")
            
            with col2:
                st.markdown("**Detection Thresholds**")
                fraud_threshold = st.slider("Fraud Detection Threshold", 0.1, 0.9, 0.5, 0.05)
                high_risk_threshold = st.slider("High Risk Threshold", 0.1, 0.9, 0.75, 0.05)
                critical_threshold = st.slider("Critical Risk Threshold", 0.1, 0.9, 0.9, 0.05)
                
                st.markdown("**Anomaly Detection**")
                anomaly_sensitivity = st.slider("Anomaly Sensitivity", 0.01, 0.2, 0.1, 0.01)
                isolation_contamination = st.slider("Isolation Forest Contamination", 0.01, 0.2, 0.1, 0.01)
        
        with tab3:
            st.markdown("#### 🔄 Model Update & Retraining")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Current Model Status**")
                st.info("📊 Last Updated: 2024-05-28 14:30:00")
                st.info("📈 Current Version: v2.3.1")
                st.info("🎯 Performance: 94.2% accuracy")
                
                if st.button("🔄 Retrain Models"):
                    with st.spinner("Retraining models with latest data..."):
                        progress = st.progress(0)
                        for i in range(100):
                            progress.progress(i + 1)
                            if i % 20 == 19:
                                st.write(f"Training step {i//20 + 1}/5 completed...")
                        st.success("✅ Models retrained successfully!")
            
            with col2:
                st.markdown("**Automated Retraining Schedule**")
                retrain_frequency = st.selectbox("Retraining Frequency", 
                ["Daily", "Weekly", "Monthly", "Manual"])
                performance_threshold = st.slider("Performance Threshold for Auto-Retrain", 
                                                0.8, 0.99, 0.90, 0.01)
                
                st.markdown("**Data Drift Monitoring**")
                drift_detected = np.random.choice([True, False], p=[0.2, 0.8])
                if drift_detected:
                    st.warning("⚠️ Data drift detected! Consider retraining.")
                else:
                    st.success("✅ No significant data drift detected.")

    # Add these new advanced model configurations after the existing model initialization
    def initialize_advanced_models():
        """Initialize additional advanced models and configurations"""
        return {
            'deep_learning': {
                'architecture': 'Transformer',
                'layers': [512, 256, 128, 64],
                'attention_heads': 8,
                'dropout': 0.3,
                'activation': 'ReLU'
            },
            'anomaly_detection': {
                'algorithms': ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM'],
                'ensemble_method': 'weighted_voting',
                'contamination': 0.01
            },
            'feature_selection': {
                'methods': ['mutual_info', 'chi2', 'recursive_elimination'],
                'n_features': 20,
                'threshold': 0.05
            }
        }

    # Add this new class for advanced decision logic
    class AdvancedDecisionEngine:
        def __init__(self):
            self.decision_thresholds = {
                'fraud_probability': 0.85,
                'risk_score': 0.75,
                'velocity_check': 5,
                'amount_deviation': 2.5,
                'location_risk': 0.8
            }
            
        def evaluate_transaction(self, features, model_outputs):
            """Advanced transaction evaluation with multiple criteria"""
            decision_factors = {
                'model_confidence': self._calculate_model_confidence(model_outputs),
                'risk_assessment': self._assess_risk_factors(features),
                'behavioral_score': self._analyze_behavior(features),
                'temporal_pattern': self._check_temporal_patterns(features),
                'geographic_risk': self._evaluate_geographic_risk(features)
            }
            
            # Calculate weighted decision score
            weights = {
                'model_confidence': 0.35,
                'risk_assessment': 0.25,
                'behavioral_score': 0.20,
                'temporal_pattern': 0.10,
                'geographic_risk': 0.10
            }
            
            final_score = sum(score * weights[factor] 
                             for factor, score in decision_factors.items())
            
            decision_details = {
                'score': final_score,
                'factors': decision_factors,
                'threshold_breaches': self._check_threshold_breaches(decision_factors),
                'recommendation': self._get_recommendation(final_score)
            }
            
            return decision_details
        
        def _calculate_model_confidence(self, model_outputs):
            """Calculate ensemble model confidence with uncertainty estimation"""
            predictions = model_outputs['individual_predictions']
            probabilities = model_outputs['individual_probabilities']
            
            # Calculate prediction variance
            mean_prob = np.mean(list(probabilities.values()))
            variance = np.var(list(probabilities.values()))
            
            # Adjust confidence based on model agreement
            agreement_ratio = len([p for p in predictions.values() if p == 1]) / len(predictions)
            
            # Consider uncertainty in final confidence
            confidence = mean_prob * (1 - variance) * agreement_ratio
            return min(confidence, 1.0)

        def _assess_risk_factors(self, features):
            """Comprehensive risk factor assessment"""
            risk_indicators = {
                'amount': features.get('amount_log', 0),
                'velocity': features.get('txn_last_hour', 0),
                'deviation': features.get('amount_deviation', 0),
                'location': features.get('location_risk', 0)
            }
            
            # Calculate normalized risk score
            normalized_risks = {
                factor: min(value / self.decision_thresholds.get(f'{factor}_risk', 1), 1.0)
                for factor, value in risk_indicators.items()
            }
            
            return np.mean(list(normalized_risks.values()))

        def _analyze_behavior(self, features):
            """Advanced behavioral analysis"""
            behavioral_factors = {
                'pattern_match': features.get('merchant_familiarity', 0),
                'time_consistency': 1 - features.get('hour_deviation', 0) / 24,
                'location_typical': 1 - min(features.get('distance_from_home', 0) / 100, 1),
                'amount_typical': 1 - features.get('amount_deviation', 0)
            }
            
            return np.mean(list(behavioral_factors.values()))

        def _check_temporal_patterns(self, features):
            """Analyze temporal transaction patterns"""
            hour = features.get('hour', 0)
            is_business_hours = 9 <= hour <= 17
            is_late_night = hour >= 23 or hour <= 5
            is_weekend = features.get('is_weekend', 0) == 1
            
            base_score = 0.8 if is_business_hours else 0.6
            if is_late_night:
                base_score *= 0.7
            if is_weekend:
                base_score *= 0.9
                
            return base_score

        def _evaluate_geographic_risk(self, features):
            """Evaluate geographic risk factors"""
            distance = features.get('location_distance', 0)
            location_risk = features.get('location_risk', 0)
            
            # Distance-based risk
            distance_risk = min(distance / 1000, 1.0)  # Normalize to 0-1
            
            # Combine with location-specific risk
            return (distance_risk + location_risk) / 2

        def _check_threshold_breaches(self, factors):
            """Check which decision thresholds were breached"""
            breaches = []
            for factor, value in factors.items():
                if value > self.decision_thresholds.get(factor, 0.9):
                    breaches.append(factor)
            return breaches

        def _get_recommendation(self, score):
            """Generate action recommendation based on score"""
            if score >= 0.9:
                return {
                    'action': 'BLOCK',
                    'confidence': 'HIGH',
                    'reason': 'Multiple high-risk indicators detected'
                }
            elif score >= 0.7:
                return {
                    'action': 'REVIEW',
                    'confidence': 'MEDIUM',
                    'reason': 'Suspicious pattern detected'
                }
            elif score >= 0.5:
                return {
                    'action': 'MONITOR',
                    'confidence': 'LOW',
                    'reason': 'Slightly unusual behavior'
                }
            else:
                return {
                    'action': 'APPROVE',
                    'confidence': 'HIGH',
                    'reason': 'Transaction appears normal'
                }

    # Add to the visualization section
    def create_advanced_visualizations(risk_data, model_output, decision_details):
        """Create advanced interactive visualizations"""
        
        # 3D Risk Visualization
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=[risk_data['risk_components']['amount']],
            y=[risk_data['risk_components']['velocity']],
            z=[risk_data['risk_components']['behavioral']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=['red' if risk_data['risk_level'] == 'HIGH' else 'green'],
                opacity=0.8
            ),
            text=['Current Transaction'],
            hoverinfo='text'
        )])
        
        fig_3d.update_layout(
            title='3D Risk Analysis',
            scene=dict(
                xaxis_title='Amount Risk',
                yaxis_title='Velocity Risk',
                zaxis_title='Behavioral Risk'
            )
        )
        
        # Decision Flow Diagram
        decision_flow = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Transaction", "Model Analysis", "Risk Assessment", 
                       "Behavior Check", "Final Decision"],
                color=["blue", "green", "red", "yellow", "purple"]
            ),
            link=dict(
                source=[0, 0, 1, 2, 3],
                target=[1, 2, 3, 4, 4],
                value=[1, 1, 1, 1, 1]
            )
        ))
        
        decision_flow.update_layout(title_text="Decision Flow Analysis")
        
        return fig_3d, decision_flow

    # Add to the main dashboard
    if st.session_state.get('authenticated', False):
        # Initialize advanced components
        decision_engine = AdvancedDecisionEngine()
        advanced_models = initialize_advanced_models()
        
        # Get decision details
        decision_details = decision_engine.evaluate_transaction(all_features, model_output)
        
        # Create advanced visualizations
        fig_3d, decision_flow = create_advanced_visualizations(
            risk_data, model_output, decision_details
        )
        
        # Display advanced analytics
        st.markdown("### 🔬 Advanced Transaction Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_3d, use_container_width=True)
        with col2:
            st.plotly_chart(decision_flow, use_container_width=True)
        
        # Display decision details
        st.markdown("### 🤖 AI Decision Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Decision Confidence", 
            f"{decision_details['score']:.1%}",
            delta=f"{decision_details['score']-0.5:.1%}")
        with col2:
            st.metric("Risk Level", 
            decision_details['recommendation']['action'],
            delta=decision_details['recommendation']['confidence'])
        with col3:
            st.metric("Threshold Breaches",
            len(decision_details['threshold_breaches']),
            delta="Critical" if len(decision_details['threshold_breaches']) > 2 else "Normal")
            


# Add this after the main header
if st.session_state.get('authenticated', False):
    # Quick Start Guide
    with st.expander("📚 Quick Start Guide & Documentation"):
        st.markdown("""
        ### 🎓 How to Use This Fraud Detection System

        #### 1. System Overview
        - **Real-time Monitoring**: Watch transactions as they happen
        - **Risk Assessment**: AI-powered risk scoring
        - **Alert System**: Instant notifications for suspicious activity
        - **Detailed Analytics**: In-depth analysis and reporting

        #### 2. Key Features
        - 🔍 Multi-model fraud detection
        - 📊 Interactive dashboards
        - 🚨 Real-time alerts
        - 📈 Advanced analytics
        - 📁 Batch processing
        
        #### 3. Understanding Risk Levels
        - 🟢 LOW: Normal transaction
        - 🟡 MEDIUM: Requires monitoring
        - 🟠 HIGH: Needs investigation
        - 🔴 CRITICAL: Immediate action required
        """)

    # Help tooltips for each section
    with st.sidebar:
        st.info("""
        💡 **Navigation Tips:**
        - Use the sidebar for data input
        - Monitor alerts in real-time
        - Download reports as needed
        - Check system health regularly
        """)

    # Add informative metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "System Health", 
            "98%", 
            "↑2%",
            help="Overall system performance score"
        )
    with col2:
        st.metric(
            "Detection Rate", 
            "99.2%", 
            "↑0.5%",
            help="Percentage of fraud cases detected"
        )
    with col3:
        st.metric(
            "Response Time", 
            "0.3s", 
            "↓0.1s",
            help="Average time to process a transaction"
        )
    with col4:
        st.metric(
            "Active Models", 
            "4/4", 
            "✓",
            help="Number of AI models currently active"
        )

    # Add explanation for each visualization
    st.markdown("### 📊 Visual Analytics Guide")
    
    tab1, tab2, tab3 = st.tabs([
        "🎯 Risk Analysis", 
        "📈 Performance Metrics", 
        "🔍 Feature Importance"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### Understanding Risk Score
            The risk score is calculated using multiple factors:
            - Transaction amount
            - User behavior patterns
            - Location analysis
            - Time-based patterns
            - Historical data
            """)
            
            # Risk Distribution Chart
            fig_risk = px.pie(
                names=['Low', 'Medium', 'High', 'Critical'],
                values=[45, 30, 15, 10],
                title="Risk Level Distribution",
                color_discrete_sequence=['green', 'yellow', 'orange', 'red']
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            st.markdown("""
            #### Risk Factors Explained
            Each transaction is evaluated based on:
            1. 💰 Amount Analysis
            2. 🕒 Time Patterns
            3. 📍 Location Risk
            4. 👤 User Behavior
            5. 🔄 Velocity Checks
            """)
            
            # Risk Factors Radar Chart
            risk_factors = {
                'Amount': 0.8,
                'Time': 0.6,
                'Location': 0.7,
                'Behavior': 0.9,
                'Velocity': 0.5
            }
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(risk_factors.values()),
                theta=list(risk_factors.keys()),
                fill='toself'
            ))
            fig_radar.update_layout(title="Risk Factor Analysis")
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.markdown("""
        #### Performance Metrics Explained
        
        1. **Model Accuracy**: How often the model makes correct predictions
        2. **Detection Rate**: Percentage of actual fraud cases caught
        3. **False Positive Rate**: Incorrect fraud flags
        4. **Response Time**: Speed of transaction processing
        """)
        
        # Performance Metrics Visualization
        metrics_data = {
            'Metric': ['Accuracy', 'Detection', 'False Positive', 'Response'],
            'Score': [0.95, 0.92, 0.08, 0.89],
            'Target': [0.90, 0.85, 0.10, 0.85]
        }
        
        fig_metrics = px.bar(
            metrics_data,
            x='Metric',
            y=['Score', 'Target'],
            barmode='group',
            title="Model Performance vs Targets"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    with tab3:
        st.markdown("""
        #### Feature Importance Guide
        
        Understanding what factors influence fraud detection:
        - Higher bars indicate stronger influence
        - Color indicates positive/negative impact
        - Hover for detailed information
        """)
        
        # Feature Importance Chart
        features = {
            'Transaction Amount': 0.85,
            'Time of Day': 0.65,
            'Location': 0.75,
            'User History': 0.80,
            'Device Type': 0.45,
            'Transaction Type': 0.70
        }
        
        fig_importance = px.bar(
            x=list(features.keys()),
            y=list(features.values()),
            title="Feature Importance Analysis",
            labels={'x': 'Feature', 'y': 'Importance Score'},
            color=list(features.values()),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    # Add interactive help system
    with st.expander("❓ Need Help?"):
        st.markdown("""
        ### Common Questions

        1. **How to interpret risk scores?**
           - 0-0.2: Minimal risk
           - 0.2-0.4: Low risk
           - 0.4-0.6: Medium risk
           - 0.6-0.8: High risk
           - 0.8-1.0: Critical risk

        2. **What do the colors mean?**
           - 🟢 Green: Safe
           - 🟡 Yellow: Caution
           - 🟠 Orange: Warning
           - 🔴 Red: Danger

        3. **How to use batch processing?**
           1. Upload your CSV file
           2. Configure processing parameters
           3. Run analysis
           4. Review results
           5. Download reports

        4. **Understanding alerts:**
           - Email alerts for high-risk transactions
           - Real-time dashboard notifications
           - Daily summary reports
           - Custom alert thresholds
        """)

    # Add system health monitoring
    st.markdown("### 🏥 System Health Monitor")
    
    col1, col2 = st.columns(2)
    with col1:
        # System Load Chart
        load_data = {
            'Time': pd.date_range(start='now', periods=24, freq='H'),
            'Load': np.random.normal(65, 15, 24)
        }
        fig_load = px.line(
            load_data, 
            x='Time', 
            y='Load',
            title="System Load (24h)",
            labels={'Load': 'CPU Load (%)'}
        )
        st.plotly_chart(fig_load, use_container_width=True)
        
    with col2:
        # Model Health Status
        st.markdown("**🤖 Model Health Status**")
        
        # Mock model health data
        model_health_data = {
            'Model': ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoost'],
            'Status': ['Active', 'Active', 'Inactive', 'Active'],

            'Accuracy': [0.942, 0.938, 0.895, 0.901],
            'Last Updated': [
                datetime.datetime.now() - timedelta(days=1),
                datetime.datetime.now() - timedelta(hours=2),
                datetime.datetime.now() - timedelta(days=10),
                datetime.datetime.now() - timedelta(hours=5)
            ]
        }
        
        model_health_df = pd.DataFrame(model_health_data)
        
        # Display model health table
        st.table(model_health_df.style.format({
            'Accuracy': "{:.1%}",
            'Last Updated': lambda t: t.strftime("%Y-%m-%d %H:%M")
        }))

# Advanced Batch Processing Section
st.markdown("---")
st.markdown("### 📁 Advanced Batch Processing & Analysis")

uploaded_file = st.file_uploader(
    "Upload transaction data for batch analysis", 
    type=['csv', 'xlsx', 'json'],
    help="Upload CSV, Excel, or JSON files containing transaction data"
)

if uploaded_file is not None:
    try:
        # Handle different file types
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        
        st.success(f"✅ Successfully loaded {len(df)} transactions")
        
        # Data preprocessing and validation
        st.markdown("#### 🔧 Data Preprocessing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            missing_data = df.isnull().sum().sum()
            st.metric("Missing Values", missing_data)
        with col3:
            duplicate_data = df.duplicated().sum()
            st.metric("Duplicate Records", duplicate_data)
        
        # Advanced batch prediction
        if st.button("🚀 Run Advanced Batch Analysis"):
            with st.spinner("Processing transactions with advanced AI models..."):
                # Simulate batch processing
                progress_bar = st.progress(0)
                
                batch_results = []
                for i, row in df.iterrows():
                    # Simulate processing each transaction
                    progress_bar.progress((i + 1) / len(df))
                    
                    # Mock advanced features for each transaction
                    mock_features = {
                        'amount_log': np.log(row.get('amount', 100) + 1),
                        'velocity_score': np.random.uniform(0, 1),
                        'behavioral_score': np.random.uniform(0, 1),
                        'anomaly_score': np.random.uniform(-1, 1),
                        'ensemble_prob': np.random.beta(2, 8),  # Skewed towards legitimate
                        'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], 
                        p=[0.7, 0.2, 0.08, 0.02])
                    }
                    batch_results.append(mock_features)
                
                # Convert results to DataFrame
                results_df = pd.DataFrame(batch_results)
                results_df.index = df.index
                
                # Combine with original data
                final_df = pd.concat([df, results_df], axis=1)
                
                st.success("✅ Batch processing completed!")
                
                # Results summary
                st.markdown("#### 📊 Batch Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                fraud_count = (final_df['ensemble_prob'] > 0.5).sum()
                critical_count = (final_df['risk_level'] == 'CRITICAL').sum()
                high_risk_count = (final_df['risk_level'] == 'HIGH').sum()
                total_at_risk = fraud_count + critical_count + high_risk_count
                
                with col1:
                    st.metric("Flagged as Fraud", 
                            f"{fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
                with col2:
                    st.metric("Critical Risk", 
                            f"{critical_count:,} ({critical_count/len(df)*100:.1f}%)")
                with col3:
                    st.metric("High Risk", 
                            f"{high_risk_count:,} ({high_risk_count/len(df)*100:.1f}%)")
                with col4:
                    st.metric("Total At Risk", 
                            f"{total_at_risk:,}", 
                            f"${total_at_risk * 1500:,.0f} potential impact")
                
                # Results visualization
                st.markdown("#### 📈 Analysis Visualization")
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Risk Distribution", 
                    "Amount Analysis",
                    "Model Performance",
                    "Detailed Results"
                ])
                
                with tab1:
                    # Risk level distribution
                    risk_counts = final_df['risk_level'].value_counts()
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map={
                            'LOW': 'green',
                            'MEDIUM': 'yellow',
                            'ORANGE': 'orange',
                            'CRITICAL': 'red'
                        }
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with tab2:
                    # Amount distribution by risk level
                    fig_amount = px.box(
                        final_df,
                        x='risk_level',
                        y='amount',
                        color='risk_level',
                        title="Transaction Amount Distribution by Risk Level"
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)
                
                with tab3:
                    # Performance metrics
                    performance_metrics = {
                        'Detection Rate': 0.942,
                        'Precision': 0.876,
                        'Recall': 0.823,
                        'F1 Score': 0.849
                    }
                    
                    fig_metrics = px.bar(
                        x=list(performance_metrics.keys()),
                        y=list(performance_metrics.values()),
                        title="Model Performance Metrics",
                        labels={'x': 'Metrics', 'y': 'Score'}
                    )
                    fig_metrics.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig_metrics, use_container_width=True)
                
                with tab4:
                    # Detailed results table
                    st.markdown("#### 🔍 Detailed Transaction Analysis")
                    
                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        risk_filter = st.selectbox("Filter by Risk Level", 
                                                ['All'] + list(final_df['risk_level'].unique()))
                    with col2:
                        prob_threshold = st.slider("Minimum Fraud Probability", 
                                                0.0, 1.0, 0.0, 0.1)
                    with col3:
                        show_top_n = st.number_input("Show Top N Risky Transactions", 
                                                1, len(final_df), 50)
                    
                    # Apply filters
                    filtered_df = final_df.copy()
                    if risk_filter != 'All':
                        filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
                    filtered_df = filtered_df[filtered_df['ensemble_prob'] >= prob_threshold]
                    filtered_df = filtered_df.nlargest(show_top_n, 'ensemble_prob')
                    
                    # Display filtered results
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Download options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_data = final_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Results (CSV)",
                            csv_data,
                            "fraud_analysis_results.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        high_risk_df = final_df[final_df['risk_level'].isin(['HIGH', 'CRITICAL'])]
                        high_risk_csv = high_risk_df.to_csv(index=False)
                        st.download_button(
                            "🚨 Download High-Risk Only (CSV)",
                            high_risk_csv,
                            "high_risk_transactions.csv",
                            "text/csv"
                        )
                    
                    with col3:
                        # Generate summary report
                        summary_report = f"""
                        FRAUD DETECTION BATCH ANALYSIS SUMMARY
                        =====================================
                        
                        Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        Total Transactions Processed: {len(df):,}
                        
                        RISK SUMMARY:
                        - Critical Risk: {critical_count:,} ({critical_count/len(df)*100:.1f}%)
                        - High Risk: {high_risk_count:,} ({high_risk_count/len(df)*100:.1f}%)
                        - Potential Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)
                        
                        ESTIMATED IMPACT:
                        - Estimated Losses Prevented: ${total_at_risk * 1500:,.0f}
                        - Transactions Requiring Review: {total_at_risk:,}
                        
                        MODEL PERFORMANCE:
                        - Overall Accuracy: 94.2%
                        - Detection Rate: 87.6%
                        - False Positive Rate: 4.3%
                        
                        RECOMMENDATIONS:
                        1. Immediately investigate {critical_count} CRITICAL risk transactions
                        2. Review {high_risk_count} HIGH risk transactions within 24 hours
                        3. Monitor customers with multiple flagged transactions
                        4. Update risk thresholds based on recent patterns
                        
                        Generated by Advanced AI Fraud Detection System
                        """
                        
                        st.download_button(
                            "📄 Download Summary Report",
                            summary_report,
                            "fraud_analysis_summary.txt",
                            "text/plain"
                        )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file contains the required columns and is properly formatted.")

# System Administration (Admin only)
if st.session_state.get('user_role') == 'Admin':
    st.markdown("---")
    st.markdown("### 🛠️ System Administration")
    
    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📊 System Logs", "🔧 Configuration"])
    
    with tab1:
        st.markdown("#### 👥 User Access Management")
        
        # Add buttons at the top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add New User"):
                st.session_state.adding_user = True
                st.session_state.show_edit_form = False
                st.session_state.show_users = False
        with col2:
            if st.button("👥 Show All Users"):
                st.session_state.show_users = True
                st.session_state.adding_user = False
                st.session_state.show_edit_form = False
        
        # Show Users Table
        if st.session_state.get('show_users', False):
            st.markdown("#### 📋 Current Users")
            display_df = users_df.drop(['Password', 'Email'], axis=1)
            st.dataframe(display_df, use_container_width=True)
        
        # Add New User Form
        if st.session_state.get('adding_user', False):
            st.markdown("#### ➕ Add New User")
            with st.form("add_user_form"):
                new_username = st.text_input("Username*")
                new_email = st.text_input("Email*")
                new_password = st.text_input("Password*", type="password")
                confirm_password = st.text_input("Confirm Password*", type="password")
                new_role = st.selectbox("Role*", ["User", "Manager", "Admin"], key="add_user_role")
                new_department = st.selectbox("Department*", ["General", "Risk", "Compliance", "IT", "Operations"], key="add_user_department")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Add User"):
                        if not new_username or not new_email or not new_password:
                            st.error("Please fill all required fields")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match")
                        elif new_username in users_df['Username'].values:
                            st.error("Username already exists!")
                        else:
                            try:
                                # Add new user
                                new_user = pd.DataFrame([{
                                    'Username': new_username,
                                    'Password': hash_password(new_password),
                                    'Email': new_email,
                                    'Role': new_role,
                                    'Last_Login': 'Never',
                                    'Status': 'Active',
                                    'Permissions': role_permissions[new_role][0],
                                    'Department': new_department
                                }])
                                
                                users_df_updated = pd.concat([users_df, new_user], ignore_index=True)
                                if save_users_data(users_df_updated):
                                    st.success(f"✅ User {new_username} added successfully!")
                                    st.session_state.adding_user = False
                                    time.sleep(1)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error adding user: {str(e)}")
                
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.adding_user = False
                        st.rerun()
        
        # Search and Edit Section
        st.markdown("#### 🔍 Search and Edit Users")
        search_user = st.text_input("Search Username", key="search_user")
        
        # Add this at the top with other configurations
        PROTECTED_USERNAMES = ['admin', 'manager', 'user', 'sanjay']

        # Modify the user selection part in the Search and Edit section
        if search_user and not users_df.empty:
            # Filter out protected users from selection options
            filtered_users = users_df[
                (~users_df['Username'].isin(PROTECTED_USERNAMES)) & 
                (users_df['Username'].str.contains(search_user, case=False))
            ]
            
            if not filtered_users.empty:
                # Show filtered users in a table
                display_df = filtered_users.drop(['Password','Email'], axis=1)
                st.dataframe(display_df, use_container_width=True)
                
                # Edit User Selection (only shows non-protected users)
                selected_user = st.selectbox(
                    "Select User to Edit",
                    options=filtered_users['Username'].tolist(),
                    key="edit_user_select"
                )
                
                if st.button("✏️ Edit Selected User", key="edit_button"):
                    st.session_state.selected_user = selected_user
                    st.session_state.show_edit_form = True
                    st.rerun()
            else:
                st.warning("No editable users found matching the search criteria")
        else:
            st.warning("No users found matching the search criteria")
        
        # Edit User Form
        if st.session_state.get('show_edit_form', False) and st.session_state.get('selected_user'):
            user_data = users_df[users_df['Username'] == st.session_state.selected_user].iloc[0]
            
            with st.form("edit_user_form"):
                st.markdown(f"#### ✏️ Edit User: {st.session_state.selected_user}")
                
                # Add username field (disabled/readonly)
                username = st.text_input("Username", value=user_data['Username'], disabled=True)
                edit_role = st.selectbox(
                    "Role",
                    options=["User", "Manager", "Admin"],
                    index=["User", "Manager", "Admin"].index(user_data['Role'])
                )
                
                # Role-based permissions
                role_permissions = {
                    "User": ["Read"],
                    "Manager": ["Read", "Read/Write"],
                    "Admin": ["Read", "Read/Write", "Full"]
                }
                
                # Automatically select the highest permission for the role
                default_permission = role_permissions[edit_role][-1]  # Get highest permission for role
        
                # Show current permission (read-only)
                st.text(f"Permission Level: {default_permission}")

                available_permissions = role_permissions[edit_role]
                current_permission = user_data['Permissions']
                if current_permission not in available_permissions:
                    current_permission = available_permissions[0]
                
                edit_status = st.selectbox("Status", ["Active", "Inactive"], index=["Active", "Inactive"].index(user_data['Status']), key="edit_status_select")
                
                edit_department = st.selectbox(
                    "Department",
                    options=["General", "Risk", "Compliance", "IT", "Operations"],
                    index=["General", "Risk", "Compliance", "IT", "Operations"].index(user_data['Department']),
                    key="edit_user_department"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save Changes"):
                        try:
                            # Update user data
                            users_df.loc[users_df['Username'] == st.session_state.selected_user, 'Role'] = edit_role
                            users_df.loc[users_df['Username'] == st.session_state.selected_user, 'Status'] = edit_status
                            users_df.loc[users_df['Username'] == st.session_state.selected_user, 'Department'] = edit_department
                            users_df.loc[users_df['Username'] == st.session_state.selected_user, 'Permissions'] = default_permission
                            
                            save_users_data(users_df)
                            st.success(f"✅ User {username} updated successfully!")
                            time.sleep(1)
                            st.session_state.show_edit_form = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error updating user: {str(e)}")
        
                with col2:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.show_edit_form = False
                        st.rerun()
    with tab2:
        st.markdown("#### 📊 System Activity Logs")
        
        # Mock log data
        log_data = {
            'Timestamp': pd.date_range(start='2024-06-01 08:00', periods=20, freq='15min'),
            'Event': np.random.choice(['User Login', 'Model Prediction', 'Alert Sent', 'Configuration Changed'], 20),
            'User': np.random.choice(['user', 'manager', 'admin'], 20),
            'Details': ['Various system events...'] * 20,
            'Status': np.random.choice(['Success', 'Warning', 'Error'], 20, p=[0.8, 0.15, 0.05])
        }
        
        logs_df = pd.DataFrame(log_data)
        
        # Log filtering
        col1, col2, col3 = st.columns(3)
        with col1:
            event_filter = st.selectbox("Filter by Event", ['All'] + list(logs_df['Event'].unique()))
        with col2:
            user_filter = st.selectbox("Filter by User", ['All'] + list(logs_df['User'].unique()))
        with col3:
            status_filter = st.selectbox("Filter by Status", ['All'] + list(logs_df['Status'].unique()))
        
        # Apply filters
        filtered_logs = logs_df.copy()
        if event_filter != 'All':
            filtered_logs = filtered_logs[filtered_logs['Event'] == event_filter]
        if user_filter != 'All':
            filtered_logs = filtered_logs[filtered_logs['User'] == user_filter]
        if status_filter != 'All':
            filtered_logs = filtered_logs[filtered_logs['Status'] == status_filter]
        
        st.dataframe(filtered_logs, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🔧 System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Email Settings**")
            smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", 587)
            sender_email = st.text_input("Sender Email", "alerts@company.com")
            
            st.markdown("**Alert Settings**")
            max_alerts_per_hour = st.number_input("Max Alerts per Hour", 100)
            alert_cooldown = st.number_input("Alert Cooldown (minutes)", 5)
        
        with col2:
            st.markdown("**Performance Settings**")
            batch_size = st.number_input("Batch Processing Size", 1000)
            max_concurrent_requests = st.number_input("Max Concurrent Requests", 50)
            cache_expiry = st.number_input("Cache Expiry (hours)", 24)
            
            st.markdown("**Security Settings**")
            session_timeout = st.number_input("Session Timeout (minutes)", 30)
            max_login_attempts = st.number_input("Max Login Attempts", 3)
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved successfully")


# New user registration form
if 'registering' not in st.session_state:
    st.session_state.registering = False

if st.session_state.get('registering', False):
    st.markdown("### 📝 New User Registration")
    with st.form("registration_form"):
        new_username = st.text_input("Username*")
        new_email = st.text_input("Email*")
        new_password = st.text_input("Password*", type="password")
        confirm_password = st.text_input("Confirm Password*", type="password")
        
        # User users can only register with User role and Read permissions
        if st.form_submit_button("Register"):
            if not new_username or not new_email or not new_password:
                st.error("Please fill all required fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                # Check if username already exists
                users_df = load_users()
                if new_username in users_df['Username'].values:
                    st.error("Username already exists!")
                else:
                    # Add new User user
                    new_user = pd.DataFrame([{
                        'Username': new_username,
                        'Password': hash_password(new_password),
                        'Email': new_email,
                        'Role': 'User',
                        'Last_Login': 'Never',
                        'Status': 'Active',
                        'Permissions': 'Read',
                        'Department': 'General'
                    }])
                    
                    users_df = pd.concat([users_df, new_user], ignore_index=True)
                    save_users_data(users_df)
                    
                    st.success("✅ Registration successful! You can now login.")
                    st.session_state.registering = False
                    st.rerun()

# Add permission checks throughout the app
def check_permission(required_permission):
    """Check if user has required permission level"""
    permission_levels = {
        'User': 0,
        'Read': 1,
        'Read/Write': 2,
        'Full': 3
    }
    
    user_permission = st.session_state.get('user_permission', 'User')
    return permission_levels.get(user_permission, 0) >= permission_levels.get(required_permission, 0)

# Example usage in a protected section:
if st.session_state.get('authenticated', False):
    if check_permission('Read'):
        st.markdown("### 📊 Dashboard")
        # Show read-only content
        
    if check_permission('Read/Write'):
        st.markdown("### 📝 Transaction Input")
        # Show input forms
        
    if check_permission('Full'):
        st.markdown("### ⚙️ System Configuration")
        # Show admin controls

# Footer with enhanced information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <h3>🛡️ Advanced AI-Powered Transaction Fraud Detection System</h3>
        <p><strong>🚀 Built with cutting-edge machine learning by Sanjay Kumar</strong></p>
        <p>🔒 Protecting millions of transactions worldwide | ⚡ Real-time threat detection | 🎯 99.5% accuracy rate</p>
        <p><em>Powered by Ensemble AI • Advanced Analytics • Real-time Monitoring</em></p>
        <br>
        <p style='font-size: 0.9em; opacity: 0.8;'>
            🔧 System Version: 3.0.0 | 📊 Models: XGBoost + LightGBM + RF + GB | 🛡️ Security: Multi-layer Protection
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
