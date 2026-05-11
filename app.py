import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score

# 1. DARK THEME UI CONFIGURATION
st.set_page_config(page_title="Retail Insights Pro", layout="wide")

# Custom CSS for high-visibility Dark Mode and Table Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 2px solid #3b82f6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 16px !important; }
    div[data-testid="stMetricValue"] { color: #3b82f6 !important; font-size: 32px !important; font-weight: bold !important; }
    
    /* Style the dataframe to be readable in dark mode */
    .stDataFrame {
        background-color: #1f2937;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Customer Loyalty Intelligence")
st.markdown("An Analysis of Frequent Shopper vs. Single-Purchase Behavior")

# 2. SIDEBAR
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Transaction Data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 3. FEATURE ENGINEERING
    df['price_variance'] = df.groupby('customer_id')['price'].transform('std').fillna(0)
    df['discount_dep'] = df['discount'] / (df['total_spend'] + 0.01)
    df['elasticity'] = (df['total_spend'] / (df['price'] + 1))
    
    features = ['price', 'discount', 'total_spend', 'price_variance', 'discount_dep', 'elasticity']
    X = df[features]; y = df['loyalty_flag']
    
    # Modeling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = PLSRegression(n_components=2)
    model.fit(X_train, y_train)
    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()

    # 4. HIGH-VISIBILITY KPI CARDS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    k2.metric("Total Customers", len(df['customer_id'].unique()))
    k3.metric("Avg Discount Dep.", f"{df['discount_dep'].mean():.2%}")
    k4.metric("Loyalty Ratio", f"{(df['loyalty_flag'].mean()):.1%}")

    st.divider()

    # 5. TABS (Including the Data Table)
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🎯 Segmentation", "📋 Data Intelligence"])

    with tab1:
        col_a, col_b = st.columns(2)
        plt.style.use('dark_background') 
        
        with col_a:
            st.subheader("Feature Impact")
            importance = pd.Series(model.x_loadings_[:, 0], index=features).sort_values()
            fig, ax = plt.subplots()
            importance.plot(kind='barh', color='#3b82f6', ax=ax)
            st.pyplot(fig)
        
        with col_b:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df[features].corr(), annot=True, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)

    with tab2:
        st.subheader("Customer Clusters (PLS Space)")
        X_pls = model.transform(X_scaled)
        fig_scat, ax_scat = plt.subplots(figsize=(10, 5))
        scatter = ax_scat.scatter(X_pls[:, 0], X_pls[:, 1], c=y, cmap='coolwarm', alpha=0.8)
        fig_scat.colorbar(scatter, ax=ax_scat, label="Loyalty Status")
        st.pyplot(fig_scat)

    with tab3:
        st.subheader("Processed Customer Data")
        st.markdown("This table shows the calculated values for price variance and discount dependence used by the model.")
        # Display the dataframe with a highlight on the loyalty flag
        st.dataframe(df.style.background_gradient(cmap='Blues', subset=['total_spend', 'discount_dep']), use_container_width=True)
        
        # Download button for the processed data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Processed Data", csv, "loyalty_analysis.csv", "text/csv")

else:
    st.info("Please upload your CSV to begin.")