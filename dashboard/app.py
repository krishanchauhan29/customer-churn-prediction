import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = pickle.load(open('src/churn_model.pkl', 'rb'))
    scaler = pickle.load(open('src/scaler.pkl', 'rb'))
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data/churn_processed.csv')

model, scaler = load_model()
df = load_data()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/crystal-ball.png", width=80)
st.sidebar.title("🔮 Churn Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict Churn"])

# ==================== DASHBOARD PAGE ====================
if page == "📊 Dashboard":
    st.title("📊 Customer Churn Analytics Dashboard")
    st.markdown("**ML-powered churn analysis with business impact insights**")
    st.markdown("---")

    # KPIs
    total = len(df)
    churned = df['Churn'].sum()
    churn_rate = churned / total * 100
    avg_charges = df['MonthlyCharges'].mean()
    revenue_risk = churned * avg_charges

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("⚠️ Churned Customers", f"{churned:,}")
    col3.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("💸 Monthly Revenue at Risk", f"${revenue_risk:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        contract_churn = df.groupby('Contract')['Churn'].mean() * 100
        contract_map = {0: 'Month-to-Month', 1: 'One Year', 2: 'Two Year'}
        contract_churn.index = [contract_map.get(i, i) for i in contract_churn.index]
        fig = px.bar(x=contract_churn.index, y=contract_churn.values,
                     title='📋 Churn Rate by Contract Type',
                     color=contract_churn.values,
                     color_continuous_scale='RdYlGn_r',
                     labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x='MonthlyCharges', color='Churn',
                           title='💰 Monthly Charges Distribution by Churn',
                           color_discrete_map={0: '#43A047', 1: '#E53935'},
                           labels={'Churn': 'Churned'},
                           barmode='overlay', opacity=0.7)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        tenure_churn = df.groupby('tenure')['Churn'].mean() * 100
        fig = px.line(x=tenure_churn.index, y=tenure_churn.values,
                      title='📅 Churn Rate by Tenure (Months)',
                      labels={'x': 'Tenure (Months)', 'y': 'Churn Rate (%)'},
                      color_discrete_sequence=['#E53935'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(df, names='Churn', title='🥧 Overall Churn Distribution',
                     color_discrete_sequence=['#43A047', '#E53935'],
                     hole=0.4)
        fig.update_traces(labels=['No Churn', 'Churned'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**📌 Key Insights:** Month-to-Month contracts have 3x higher churn | New customers (0-1yr) churn most | Higher charges = higher churn risk | Fiber optic users churn more than DSL")

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Predict Churn":
    st.title("🔮 Customer Churn Predictor")
    st.markdown("**Enter customer details to predict churn probability**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Contract Info")
        contract = st.selectbox("Contract Type", [0, 1, 2],
                                 format_func=lambda x: ['Month-to-Month', 'One Year', 'Two Year'][x])
        internet = st.selectbox("Internet Service", [0, 1, 2],
                                 format_func=lambda x: ['DSL', 'Fiber Optic', 'No'][x])
        payment = st.selectbox("Payment Method", [0, 1, 2, 3],
                                format_func=lambda x: ['Bank Transfer', 'Credit Card', 'Electronic Check', 'Mailed Check'][x])

    with col2:
        st.subheader("💰 Charges Info")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 119, 65)
        total_charges = st.slider("Total Charges ($)", 18, 8685, 1500)

    with col3:
        st.subheader("🛠️ Services")
        phone = st.selectbox("Phone Service", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
        paperless = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: ['No', 'Yes'][x])

    st.markdown("---")

    if st.button("🔮 Predict Churn", use_container_width=True):
        input_data = np.zeros(19)
        input_data[1] = senior
        input_data[4] = tenure
        input_data[5] = phone
        input_data[7] = internet
        input_data[14] = contract
        input_data[15] = paperless
        input_data[16] = payment
        input_data[17] = monthly_charges
        input_data[18] = total_charges

        input_scaled = scaler.transform([input_data])
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]

        col1, col2, col3 = st.columns(3)

        if prediction == 1:
            col2.error(f"⚠️ HIGH CHURN RISK: {prob*100:.1f}%")
            st.warning("**Recommended Actions:** Offer loyalty discount | Switch to annual contract | Assign dedicated support")
        else:
            col2.success(f"✅ LOW CHURN RISK: {prob*100:.1f}%")
            st.info("**Customer is likely to stay.** Consider upselling premium services.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E53935" if prob > 0.5 else "#43A047"},
                'steps': [
                    {'range': [0, 30], 'color': '#E8F5E9'},
                    {'range': [30, 70], 'color': '#FFF9C4'},
                    {'range': [70, 100], 'color': '#FFEBEE'}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

st.caption("Built by Krishan Kumar Chauhan | M.Tech Data Science, GBU")
