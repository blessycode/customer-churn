import streamlit as st
import pandas as pd
import pickle
import warnings
import os
from datetime import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="📱 Telco Customer Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data storage file
DATA_STORAGE_FILE = "customer_predictions_log.csv"

# Custom CSS with professional dark mode design
st.markdown("""
    <style>
    /* Global Dark Mode Theme */
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --bg-input: #0f172a;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border-color: #334155;
        --accent-primary: #3b82f6;
        --accent-secondary: #8b5cf6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main Background */
    .main {
        background-color: var(--bg-primary);
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 40px 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Input Sections */
    .input-section {
        background: var(--bg-card);
        padding: 32px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .input-section:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 2px solid var(--accent-primary);
        letter-spacing: -0.01em;
    }
    
    /* Prediction Results */
    .prediction-result-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 32px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-result-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 32px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-icon {
        font-size: 3.5rem;
        margin-bottom: 12px;
    }
    
    .prediction-label {
        font-size: 1rem;
        opacity: 0.95;
        margin-bottom: 8px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .prediction-value {
        font-size: 2.75rem;
        font-weight: 900;
        margin-top: 12px;
        letter-spacing: -0.02em;
    }
    
    /* Streamlit Component Styling */
    .stSelectbox > div > div {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }
    
    .stSlider > div > div {
        background-color: var(--bg-input) !important;
    }
    
    .stSlider label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
        color: white !important;
        border-color: var(--accent-primary) !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }
    
    /* Text Styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    
    p, div, span {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
    }
    
    /* Info Boxes */
    .stInfo {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-primary);
    }
    
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: var(--success) !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: var(--warning) !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: var(--danger) !important;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: var(--text-muted) !important;
        font-size: 0.875rem;
        padding: 20px 0;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'model.pkl' not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Prepare input data to match model's expected format
def prepare_input_data(input_dict):
    """
    Prepare input data to match the model's expected feature format.
    Based on the training notebook: prepare_telco_df function
    """
    # Start with a dictionary for the encoded features
    encoded = {}
    
    # Binary encoding: Gender (Male=0, Female=1)
    encoded['gender'] = 1 if input_dict['gender'] == "Female" else 0
    
    # Binary encoding: SeniorCitizen (No=0, Yes=1)
    encoded['SeniorCitizen'] = 1 if input_dict['SeniorCitizen'] == "Yes" else 0
    
    # Binary encoding: Partner, Dependents, PhoneService, PaperlessBilling (Yes=1, No=0)
    encoded['Partner'] = 1 if input_dict['Partner'] == "Yes" else 0
    encoded['Dependents'] = 1 if input_dict['Dependents'] == "Yes" else 0
    encoded['PhoneService'] = 1 if input_dict['PhoneService'] == "Yes" else 0
    encoded['PaperlessBilling'] = 1 if input_dict['PaperlessBilling'] == "Yes" else 0
    
    # Numeric features
    encoded['tenure'] = input_dict['tenure']
    encoded['MonthlyCharges'] = float(input_dict['MonthlyCharges'])
    # Convert TotalCharges to numeric (handle potential string values)
    try:
        encoded['TotalCharges'] = float(input_dict['TotalCharges'])
    except (ValueError, TypeError):
        encoded['TotalCharges'] = 0.0
    
    # MultipleLines encoding (No phone service=0, No=0, Yes=1)
    if input_dict['MultipleLines'] == "Yes":
        encoded['MultipleLines'] = 1
    else:
        encoded['MultipleLines'] = 0
    
    # Service columns: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    # Yes=1, No=0, "No internet service"=0
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        encoded[col] = 1 if input_dict[col] == "Yes" else 0
    
    # One-hot encoding with drop_first=True
    # InternetService: drops "DSL" (first category alphabetically)
    # So we need: InternetService_Fiber optic, InternetService_No
    encoded['InternetService_Fiber optic'] = 1 if input_dict['InternetService'] == "Fiber optic" else 0
    encoded['InternetService_No'] = 1 if input_dict['InternetService'] == "No" else 0
    
    # Contract: drops "Month-to-month" (first category alphabetically)
    # So we need: Contract_One year, Contract_Two year
    encoded['Contract_One year'] = 1 if input_dict['Contract'] == "One year" else 0
    encoded['Contract_Two year'] = 1 if input_dict['Contract'] == "Two year" else 0
    
    # PaymentMethod: drops "Bank transfer (automatic)" (first category alphabetically)
    # So we need: PaymentMethod_Credit card (automatic), PaymentMethod_Electronic check, PaymentMethod_Mailed check
    encoded['PaymentMethod_Credit card (automatic)'] = 1 if input_dict['PaymentMethod'] == "Credit card (automatic)" else 0
    encoded['PaymentMethod_Electronic check'] = 1 if input_dict['PaymentMethod'] == "Electronic check" else 0
    encoded['PaymentMethod_Mailed check'] = 1 if input_dict['PaymentMethod'] == "Mailed check" else 0
    
    # Convert to DataFrame with correct column order (matching model's expected order)
    # Order matches the notebook output exactly (excluding Churn target)
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'InternetService_No',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Create DataFrame with all features in order
    input_df = pd.DataFrame([encoded])
    
    # Ensure all columns are present (fill missing with 0)
    for feature in feature_order:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns to match expected order
    input_df = input_df[feature_order]
    
    return input_df

# Save prediction to log file
def save_prediction(input_dict, prediction, churn_probability):
    """Save prediction data to CSV file"""
    # Define columns for storage (original values, not encoded)
    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'gender': input_dict['gender'],
        'SeniorCitizen': input_dict['SeniorCitizen'],
        'Partner': input_dict['Partner'],
        'Dependents': input_dict['Dependents'],
        'tenure': input_dict['tenure'],
        'PhoneService': input_dict['PhoneService'],
        'MultipleLines': input_dict['MultipleLines'],
        'InternetService': input_dict['InternetService'],
        'OnlineSecurity': input_dict['OnlineSecurity'],
        'OnlineBackup': input_dict['OnlineBackup'],
        'DeviceProtection': input_dict['DeviceProtection'],
        'TechSupport': input_dict['TechSupport'],
        'StreamingTV': input_dict['StreamingTV'],
        'StreamingMovies': input_dict['StreamingMovies'],
        'Contract': input_dict['Contract'],
        'PaperlessBilling': input_dict['PaperlessBilling'],
        'PaymentMethod': input_dict['PaymentMethod'],
        'MonthlyCharges': input_dict['MonthlyCharges'],
        'TotalCharges': input_dict['TotalCharges'],
        'prediction': int(prediction),
        'churn_probability': round(churn_probability, 4)
    }
    
    # Create or append to CSV
    if os.path.exists(DATA_STORAGE_FILE):
        df_existing = pd.read_csv(DATA_STORAGE_FILE)
        df_new = pd.DataFrame([record])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(DATA_STORAGE_FILE, index=False)
    else:
        df_new = pd.DataFrame([record])
        df_new.to_csv(DATA_STORAGE_FILE, index=False)

# Load prediction log
def load_prediction_log():
    """Load prediction log from CSV file"""
    if os.path.exists(DATA_STORAGE_FILE):
        try:
            df = pd.read_csv(DATA_STORAGE_FILE)
            return df
        except Exception as e:
            st.warning(f"Error loading prediction log: {str(e)}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# Header
st.markdown("""
    <div class='header-container'>
        <div class='header-title'>📱 Churn Predictor</div>
        <div class='header-subtitle'>Advanced ML-powered customer churn prediction with real-time monitoring</div>
    </div>
""", unsafe_allow_html=True)

# Load model
model = load_model()
if model is None:
    st.stop()

# Sidebar with app information
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **Telco Customer Churn Predictor**
    
    This application uses machine learning to predict customer churn probability for telecommunications companies. 
    
    The model analyzes customer demographics, service usage, contract details, and billing information to identify 
    customers at risk of leaving.
    """)
    
    st.markdown("---")
    st.markdown("### 🤖 Model Information")
    st.markdown("""
    **Algorithm:** Random Forest Classifier
    
    **Features Analyzed:**
    - Customer demographics
    - Service subscriptions
    - Account tenure & charges
    - Contract type
    - Payment methods
    
    **Performance:**
    - High accuracy prediction
    - Real-time risk assessment
    - Probability scoring
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    df_log = load_prediction_log()
    if len(df_log) > 0:
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Predictions", len(df_log))
        with col_stat2:
            if 'churn_probability' in df_log.columns:
                avg_prob = df_log['churn_probability'].mean() * 100
                st.metric("Avg Risk", f"{avg_prob:.1f}%")
    else:
        st.info("📊 No predictions yet")
    
    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Fill Customer Details**
       - Enter demographics and services
       - Set account information
       - Configure billing options
    
    2. **Make Prediction**
       - Click "Predict Churn Risk"
       - Review probability scores
       - Check risk level assessment
    
    3. **Analyze Results**
       - View probability distribution
       - Check risk categorization
       - Review recommendations
    """)
    
    st.markdown("---")
    st.markdown("### 🔑 Key Features")
    st.markdown("""
    ✓ **Real-time Predictions**
    
    ✓ **Risk Level Assessment**
      - Critical, High, Medium, Low
    
    ✓ **Performance Dashboard**
      - Metrics & trends
      - Visual analytics
    
    ✓ **Data Export**
      - CSV download
      - Historical tracking
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    **High Risk Indicators:**
    - Month-to-month contracts
    - High monthly charges
    - Short tenure
    - Paperless billing
    
    **Retention Factors:**
    - Long-term contracts
    - Multiple services
    - Loyalty programs
    """)
    
    st.markdown("---")
    st.markdown("### 📱 Version")
    st.caption("Version 1.0.0 | Built with Streamlit & scikit-learn")

# Tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Performance Dashboard", "📈 Data Analytics", "⚙️ Settings"])

# ============= TAB 1: PREDICTION =============
with tab1:
    st.markdown("<div class='input-section'><div class='section-title'>👤 Customer Information</div>", unsafe_allow_html=True)
    
    # Demographics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
    with col4:
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Services Section
    st.markdown("<div class='input-section'><div class='section-title'>📞 Services</div>", unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with col_s2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col_s3:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add-ons Section
    st.markdown("<div class='input-section'><div class='section-title'>🛡️ Add-on Services</div>", unsafe_allow_html=True)
    
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with col_a2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col_a3:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col_a4:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Entertainment Section
    st.markdown("<div class='input-section'><div class='section-title'>🎬 Entertainment & Billing</div>", unsafe_allow_html=True)
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    with col_e2:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    with col_e3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col_e4:
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                                      "Credit card (automatic)"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Account Details
    st.markdown("<div class='input-section'><div class='section-title'>💰 Account Details</div>", unsafe_allow_html=True)
    
    col_ac1, col_ac2, col_ac3 = st.columns(3)
    with col_ac1:
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long customer has been with company")
    with col_ac2:
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0)
    with col_ac3:
        total_charges = st.slider("Total Charges ($)", 0.0, 10000.0, 500.0)
    
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction Button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if st.button("🎯 PREDICT CHURN RISK", use_container_width=True):
            # Create input dictionary with raw values
            input_dict = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            try:
                # Prepare input
                input_df = prepare_input_data(input_dict)
                
                # Get prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Save to database with original values
                save_prediction(input_dict, prediction, prediction_proba[1])
                
                # Display results
                st.markdown("---")
                
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                
                with result_col1:
                    st.markdown("### Risk Profile")
                    if prediction == 1:
                        st.markdown(f"""
                        <div class='prediction-result-danger'>
                            <div class='prediction-icon'>⚠️</div>
                            <div class='prediction-label'>CHURN RISK</div>
                            <div class='prediction-value'>{prediction_proba[1]*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='prediction-result-success'>
                            <div class='prediction-icon'>✅</div>
                            <div class='prediction-label'>SAFE</div>
                            <div class='prediction-value'>{prediction_proba[0]*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown("### Probability Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
                    colors = ['#2ecc71', '#e74c3c']
                    bars = ax.barh(['Retention', 'Churn'], [prediction_proba[0], prediction_proba[1]], color=colors, edgecolor='black', linewidth=2)
                    ax.set_xlabel('Probability', fontweight='bold', fontsize=11)
                    ax.set_xlim([0, 1])
                    
                    for i, (bar, val) in enumerate(zip(bars, [prediction_proba[0], prediction_proba[1]])):
                        ax.text(val + 0.02, i, f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=12)
                    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with result_col3:
                    st.markdown("### Summary")
                    churn_prob = prediction_proba[1]
                    if churn_prob > 0.7:
                        st.metric("Risk Level", "🔴 CRITICAL", delta=f"{churn_prob*100:.1f}%")
                        st.error("Immediate intervention needed")
                    elif churn_prob > 0.5:
                        st.metric("Risk Level", "🟠 HIGH", delta=f"{churn_prob*100:.1f}%")
                        st.warning("Proactive retention recommended")
                    elif churn_prob > 0.3:
                        st.metric("Risk Level", "🟡 MEDIUM", delta=f"{churn_prob*100:.1f}%")
                        st.info("Monitor engagement closely")
                    else:
                        st.metric("Risk Level", "🟢 LOW", delta=f"{churn_prob*100:.1f}%")
                        st.success("Customer likely to stay")
                
                st.success("✅ Prediction saved to database!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ============= TAB 2: PERFORMANCE DASHBOARD =============
with tab2:
    st.subheader("📊 Model Performance Metrics")
    
    df_log = load_prediction_log()
    
    if len(df_log) > 0:
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Total Predictions", len(df_log))
        
        with col_m2:
            churn_rate = (df_log['prediction'].sum() / len(df_log) * 100) if 'prediction' in df_log.columns else 0
            st.metric("Churn Rate (Predicted)", f"{churn_rate:.1f}%")
        
        with col_m3:
            avg_prob = df_log['churn_probability'].mean() * 100 if 'churn_probability' in df_log.columns else 0
            st.metric("Avg Churn Probability", f"{avg_prob:.1f}%")
        
        with col_m4:
            if 'timestamp' in df_log.columns and len(df_log) > 0:
                last_update = str(df_log['timestamp'].iloc[-1])[:10] if len(df_log) > 0 else "N/A"
                st.metric("Last Updated", last_update)
            else:
                st.metric("Last Updated", "N/A")
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 Churn Probability Distribution")
            if 'churn_probability' in df_log.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df_log['churn_probability'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
                ax.axvline(df_log['churn_probability'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                ax.set_xlabel('Churn Probability', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title('Distribution of Predicted Churn Probabilities', fontweight='bold', fontsize=12)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        
        with col_chart2:
            st.subheader("🎯 Prediction Distribution")
            if 'prediction' in df_log.columns:
                pred_counts = df_log['prediction'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#2ecc71', '#e74c3c']
                labels = ['Retained', 'Churn']
                ax.pie([pred_counts.get(0, 0), pred_counts.get(1, 0)], labels=labels, autopct='%1.1f%%',
                       colors=colors, startangle=90, textprops={'fontweight': 'bold', 'fontsize': 12})
                ax.set_title('Prediction Breakdown', fontweight='bold', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Trends over time
        if 'timestamp' in df_log.columns and 'churn_probability' in df_log.columns:
            st.subheader("📊 Churn Probability Trend Over Time")
            df_log_copy = df_log.copy()
            df_log_copy['timestamp'] = pd.to_datetime(df_log_copy['timestamp'], errors='coerce')
            df_log_copy = df_log_copy.dropna(subset=['timestamp'])
            df_log_sorted = df_log_copy.sort_values('timestamp')
            
            if len(df_log_sorted) > 0:
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df_log_sorted['timestamp'], df_log_sorted['churn_probability'], marker='o', linestyle='-', 
                       linewidth=2, markersize=6, color='#9b59b6', label='Churn Probability')
                ax.fill_between(df_log_sorted['timestamp'], df_log_sorted['churn_probability'], alpha=0.3, color='#9b59b6')
                ax.set_xlabel('Date', fontweight='bold')
                ax.set_ylabel('Churn Probability', fontweight='bold')
                ax.set_title('Churn Probability Trend', fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("📊 No predictions yet. Make some predictions to see the dashboard!")

# ============= TAB 3: DATA ANALYTICS =============
with tab3:
    st.subheader("📈 Data Analytics & Insights")
    
    df_log = load_prediction_log()
    
    if len(df_log) > 10:
        col_ana1, col_ana2 = st.columns(2)
        
        with col_ana1:
            st.subheader("👥 Churn by Demographics")
            
            if 'gender' in df_log.columns and 'prediction' in df_log.columns:
                churn_by_gender = df_log.groupby('gender')['prediction'].agg(['sum', 'count'])
                churn_by_gender['churn_rate'] = (churn_by_gender['sum'] / churn_by_gender['count'] * 100)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                churn_by_gender['churn_rate'].plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
                ax.set_ylabel('Churn Rate (%)', fontweight='bold')
                ax.set_xlabel('Gender', fontweight='bold')
                ax.set_title('Churn Rate by Gender', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
        
        with col_ana2:
            st.subheader("📞 Churn by Internet Service")
            
            if 'InternetService' in df_log.columns and 'prediction' in df_log.columns:
                churn_by_internet = df_log.groupby('InternetService')['prediction'].agg(['sum', 'count'])
                churn_by_internet['churn_rate'] = (churn_by_internet['sum'] / churn_by_internet['count'] * 100)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                churn_by_internet['churn_rate'].plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#f39c12'])
                ax.set_ylabel('Churn Rate (%)', fontweight='bold')
                ax.set_xlabel('Internet Service', fontweight='bold')
                ax.set_title('Churn Rate by Internet Service', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("---")
        
        col_ana3, col_ana4 = st.columns(2)
        
        with col_ana3:
            st.subheader("📋 Churn by Contract Type")
            
            if 'Contract' in df_log.columns and 'prediction' in df_log.columns:
                churn_by_contract = df_log.groupby('Contract')['prediction'].agg(['sum', 'count'])
                churn_by_contract['churn_rate'] = (churn_by_contract['sum'] / churn_by_contract['count'] * 100)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                churn_by_contract['churn_rate'].plot(kind='barh', ax=ax, color=['#e74c3c', '#f39c12', '#2ecc71'])
                ax.set_xlabel('Churn Rate (%)', fontweight='bold')
                ax.set_title('Churn Rate by Contract', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        
        with col_ana4:
            st.subheader("💰 Churn by Tenure Groups")
            
            if 'tenure' in df_log.columns and 'prediction' in df_log.columns:
                df_log_copy = df_log.copy()
                df_log_copy['tenure_group'] = pd.cut(df_log_copy['tenure'], bins=[0, 12, 24, 48, 72], 
                                                    labels=['0-12m', '12-24m', '24-48m', '48-72m'])
                churn_by_tenure = df_log_copy.groupby('tenure_group')['prediction'].agg(['sum', 'count'])
                churn_by_tenure['churn_rate'] = (churn_by_tenure['sum'] / churn_by_tenure['count'] * 100)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                churn_by_tenure['churn_rate'].plot(kind='bar', ax=ax, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
                ax.set_ylabel('Churn Rate (%)', fontweight='bold')
                ax.set_xlabel('Tenure Group', fontweight='bold')
                ax.set_title('Churn Rate by Tenure', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
        
    else:
        st.info("📈 Need at least 10 predictions to generate analytics!")

# ============= TAB 4: SETTINGS =============
with tab4:
    st.subheader("⚙️ Settings & Data Management")
    
    df_log = load_prediction_log()
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.subheader("📊 Database Status")
        st.metric("Records in Database", len(df_log))
        if len(df_log) > 0 and 'timestamp' in df_log.columns:
            try:
                date_min = str(df_log['timestamp'].min())[:10]
                date_max = str(df_log['timestamp'].max())[:10]
                st.metric("Date Range", f"{date_min} to {date_max}")
            except:
                st.metric("Date Range", "N/A")
    
    with col_set2:
        st.subheader("📥 Export Data")
        
        if len(df_log) > 0:
            csv = df_log.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Log",
                data=csv,
                file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data to export yet.")
    
    st.markdown("---")
    st.subheader("🔄 Model Retraining")
    
    st.info("""
    **How Continuous Learning Works:**
    1. ✅ Data is automatically saved after each prediction
    2. ✅ All predictions are stored in `customer_predictions_log.csv`
    3. 📊 Dashboard monitors patterns and trends
    4. 🔄 To retrain: Export data, add actual churn outcomes, retrain in notebook
    
    **Current Status:** Model learns from accumulated predictions
    """)
    
    if len(df_log) > 0:
        st.success(f"✅ {len(df_log)} predictions logged and ready for analysis!")
    
    st.markdown("---")
    st.subheader("🗑️ Data Management")
    
    if st.button("🗑️ Clear All Prediction Data"):
        if os.path.exists(DATA_STORAGE_FILE):
            os.remove(DATA_STORAGE_FILE)
            st.success("✅ Data cleared!")
            st.rerun()
    
    st.warning("⚠️ This action cannot be undone. Make sure to export your data first!")

# Footer
st.markdown("---")
st.markdown("""
    <div class='footer-text'>
    📱 Telco Customer Churn Prediction System | Built with Streamlit & scikit-learn
    </div>
""", unsafe_allow_html=True)
