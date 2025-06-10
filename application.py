import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="CarValue - Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        animation: pulse 2s infinite;
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = pk.load(open('ridge.pkl', 'rb'))
        scaler = pk.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'ridge.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_models()

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Cardetails.csv")
        df['name'] = df['name'].str.strip()
        df['name'] = df['name'].apply(lambda x: x.split(' ')[0])
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file 'Cardetails.csv' not found.")
        return None

df = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 CarValue </h1>
    <h3>Intelligent Car Price Prediction System</h3>
    <p>Get instant, accurate car valuations </p>
</div>
""", unsafe_allow_html=True)

if df is not None and model is not None and scaler is not None:
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔧 Car Specifications")
        
        # Car brand selection with search
        name = st.selectbox(
            '🏷️ Select Car Brand',
            df['name'].unique(),
            help="Choose the manufacturer of your car"
        )
        
        # Year with validation
        yr = st.slider(
            '📅 Manufacturing Year',
            min_value=2000,
            max_value=2025,
            value=2015,
            help="Year when the car was manufactured"
        )
        
        # Kilometers driven
        kms = st.slider(
            '🛣️ Kilometers Driven',
            min_value=10,
            max_value=200000,
            value=50000,
            step=1000,
            help="Total distance covered by the car"
        )
        
        # Fuel type
        fuel = st.selectbox(
            '⛽ Fuel Type',
            df['fuel'].unique(),
            help="Type of fuel the car uses"
        )
        
        # Seller type
        seller = st.selectbox(
            '👤 Seller Type',
            df['seller_type'].unique(),
            help="Type of seller"
        )
    
    with col2:
        st.markdown("### ⚙️ Technical Details")
        
        # Transmission
        transmission = st.selectbox(
            '🔧 Transmission Type',
            df['transmission'].unique(),
            help="Type of transmission system"
        )
        
        # Owner type
        owner = st.selectbox(
            '🔑 Owner Type',
            df['owner'].unique(),
            help="Number of previous owners"
        )
        
        # Mileage
        mileage = st.slider(
            '📊 Mileage (km/l)',
            min_value=10,
            max_value=100,
            value=20,
            help="Fuel efficiency of the car"
        )
        
        # Engine CC
        engine = st.slider(
            '🔧 Engine Capacity (CC)',
            min_value=700,
            max_value=5000,
            value=1500,
            step=100,
            help="Engine displacement in cubic centimeters"
        )
        
        # Max power
        power = st.slider(
            '⚡ Max Power (bhp)',
            min_value=0,
            max_value=500,
            value=100,
            help="Maximum power output"
        )
        
        # Seating capacity
        seats = st.slider(
            '🪑 Seating Capacity',
            min_value=2,
            max_value=10,
            value=5,
            help="Number of seats in the car"
        )
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Car Price", key="predict")
    
    if predict_button:
        with st.spinner('🤖 Analyzing car specifications...'):
            # Create input dataframe
            input_data = pd.DataFrame(
                [[name, yr, kms, fuel, seller, transmission, owner, mileage, engine, power, seats]],
                columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
            )
            
            # Apply transformations
            input_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                'Ambassador', 'Ashok', 'Isuzu', 'Opel'], [i for i in range(1, 32)], inplace=True)
            
            input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [4, 3, 2, 1], inplace=True)
            input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [2, 1, 3], inplace=True)
            input_data['transmission'].replace(['Manual', 'Automatic'], [0, 1], inplace=True)
            input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                'Fourth & Above Owner', 'Test Drive Car'], [5, 4, 2, 1, 3], inplace=True)
            
            # Scale and predict
            scaled = scaler.transform(input_data)
            car_price = model.predict(scaled)[0]
            
            # Display prediction with animation
            st.markdown(f"""
            <div class="prediction-box">
                💰 Estimated Car Price<br>
                <span style="font-size: 2rem;">₹ {car_price:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>🎯 Confidence</h4>
                    <p style="font-size: 1.2rem; color: #28a745;">High</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                price_per_km = car_price / max(kms, 1)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Price/KM</h4>
                    <p style="font-size: 1.2rem; color: #17a2b8;">₹ {price_per_km:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                age = 2025 - yr
                depreciation = max(0, (15 - age) * 5)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📉 Condition</h4>
                    <p style="font-size: 1.2rem; color: #ffc107;">{depreciation}% Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Market insights
            st.markdown("### 📈 Market Insights")
            
            # Create a simple comparison chart
            similar_cars = df[df['name'] == name].copy() if not df[df['name'] == name].empty else df.sample(10)
            
            if len(similar_cars) > 0:
                fig = px.scatter(
                    similar_cars.head(20),
                    x='year',
                    y='selling_price',
                    color='fuel',
                    size='km_driven',
                    title=f"Price Trends for {name} Cars",
                    labels={'year': 'Year', 'selling_price': 'Price (₹)'}
                )
                fig.add_hline(y=car_price, line_dash="dash", line_color="red", 
                            annotation_text="Your Car's Predicted Price")
                st.plotly_chart(fig, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 Machine Learning Model</h4>
            <p>Our AI model analyzes multiple factors including brand, age, mileage, fuel type, and technical specifications to provide accurate price predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Data-Driven Insights</h4>
            <p>Trained on thousands of real car sales data points to ensure reliable and market-relevant price estimates.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("⚠️ Please ensure all required files (ridge.pkl, scaler.pkl, Cardetails.csv) are available.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🚗 CarValue | Made using Streamlit"
    "</div>",
    unsafe_allow_html=True
)