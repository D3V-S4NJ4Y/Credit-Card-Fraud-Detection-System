# 💳 Credit Card & Transaction Fraud Detection System

## Overview
A sophisticated real-time fraud detection system using ensemble machine learning models and advanced analytics to protect financial transactions.

### 🌟 Key Features
- **Real-time Fraud Detection**: Instant transaction monitoring and analysis
- **Multi-Model Ensemble**: XGBoost, LightGBM, Random Forest, and Gradient Boosting
- **Advanced Analytics Dashboard**: Interactive visualizations and insights
- **Role-Based Access Control**: User, Analyst, Manager, and Admin roles
- **Email Alert System**: Automated notifications for suspicious activities
- **Batch Processing**: Handle multiple transactions simultaneously

## 🚀 Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Models**: 
  - XGBoost
  - LightGBM
  - Random Forest
  - Gradient Boosting
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Security**: SHA-256 encryption

## 📊 Features In Detail

### 1. Fraud Detection
- Real-time transaction scoring
- Risk level assessment (LOW, MEDIUM, HIGH, CRITICAL)
- Behavioral analysis
- Location-based risk assessment
- Velocity checks

### 2. User Interface
- Interactive dashboards
- Real-time monitoring
- Advanced visualization
- Risk score breakdown
- Transaction patterns analysis

### 3. Security Features
- Role-based access control
- Password encryption
- Session management
- Activity logging
- Audit trails

### 4. Alert System
- Email notifications
- Risk-based alerting
- Customizable thresholds
- Batch alert processing

## 🚀 Dataset
The dataset used for this project is the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset?resource=download&select=credit_card_transactions.csv)
dataset from Kaggle.

## 📁 Project Structure
```
Credit-Card-Fraud-Detection/
├── 📁 data/                      # Data directory
│   ├── users.csv                 # User credentials and roles
│   ├── transactions.csv          # Dataset of credit card transactions
│   └── models/                   # ML model implementations
│       ├── xgb_model.json       # XGBoost model file
│       └── features.pkl         # Saved feature configurations
│
├── 📁 .streamlit/               # Streamlit configurations
│   ├── config.toml             # Streamlit settings
│   └── secrets.toml            # Secure credentials
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Project dependencies
├── 📄 .env                      # Environment variables
├── 📄 .gitignore               # Git ignore patterns
├── 📄 LICENSE                   # License information
└── 📄 README.md                # Project documentation
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
streamlit run app.py
```

## 👥 User Roles

1. **User**
   - View basic transaction details
   - Submit transactions
   - Receive alerts

2. **Analyst**
   - Basic analysis
   - View detailed reports
   - Monitor transactions

3. **Manager**
   - Advanced analysis
   - Team management
   - Performance monitoring

4. **Admin**
   - Full system access
   - User management
   - System configuration

## 📈 Analytics Features

### Transaction Analysis
- Risk scoring
- Behavioral patterns
- Anomaly detection
- Historical analysis

### Visualization
- 3D risk analysis
- Decision flow diagrams
- Risk component breakdown
- Performance metrics

### Reporting
- Real-time dashboards
- Batch analysis reports
- Custom filters
- Export capabilities

## ⚙️ Configuration

### Email Setup
```python
EMAIL_CONFIG = {
    "smtp_user": "your-email@gmail.com",
    "smtp_password": "your-app-password",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}
```

### Risk Thresholds
```python
RISK_THRESHOLDS = {
    'CRITICAL': 0.9,
    'HIGH': 0.7,
    'MEDIUM': 0.5,
    'LOW': 0.2
}
```

## 🔒 Security Best Practices

1. Use environment variables for sensitive data
2. Regular password updates
3. Session timeout implementation
4. Activity logging
5. Data encryption

## 📝 Logs and Monitoring

- System health monitoring
- User activity logs
- Model performance tracking
- Error logging

## 📄 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 👨‍💻 Author
**Sanjay Kumar**
- Email: sanjay.dev925@gmail.com
- GitHub: [Your GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments
- Machine Learning libraries contributors
- Streamlit community
- Security framework developers

## 📞 Support
For support and queries:
- Create an issue in the repository
- Contact: sanjay.dev925@gmail.com

---
⭐ Star this repository if you find it helpful!