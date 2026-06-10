import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(
    page_title="Nassau Candy — Factory Optimizer",
    page_icon="🍬", layout="wide"
)

@st.cache_data
def load_data():
    df   = pd.read_csv('nassau_clean.csv')
    recs = pd.read_csv('recommendations.csv')
    return df, recs

@st.cache_resource
def load_model():
    model    = joblib.load('best_model.pkl')
    features = joblib.load('features.pkl')
    scaler   = joblib.load('scaler.pkl')
    return model, features, scaler

df, recs                = load_data()
model, features, scaler = load_model()

le_f = LabelEncoder().fit(df['Factory'])
le_r = LabelEncoder().fit(df['Region'])
le_s = LabelEncoder().fit(df['Ship Mode'])
le_p = LabelEncoder().fit(df['Product Name'])
le_d = LabelEncoder().fit(df['Division'])

all_factories = sorted(df['Factory'].unique().tolist())

st.sidebar.title("🍬 Nassau Candy")
st.sidebar.markdown("Factory Optimization System")
page = st.sidebar.radio("Go to", [
    "🏭 Factory Optimizer",
    "🔄 What-If Simulator",
    "⭐ Recommendations",
    "⚠️ Risk Panel"
])

if page == "🏭 Factory Optimizer":
    st.title("🏭 Factory Optimizer")
    st.markdown("Select a product to see predicted lead time across all factories.")
    product  = st.selectbox("Select Product", sorted(df['Product Name'].unique()))
    prod_df  = df[df['Product Name'] == product]
    division = prod_df['Division'].iloc[0]
    avg_sale = prod_df['Sales'].mean()
    avg_cost = prod_df['Cost'].mean()
    avg_unit = prod_df['Units'].mean()
    avg_reg  = prod_df['Region'].mode()[0]
    avg_ship = prod_df['Ship Mode'].mode()[0]
    curr_fac = prod_df['Factory'].iloc[0]
    results  = []
    for fac in all_factories:
        scaled = scaler.transform([[avg_sale, avg_cost, avg_unit]])[0]
        X = pd.DataFrame([{
            'Factory Enc'  : le_f.transform([fac])[0],
            'Region Enc'   : le_r.transform([avg_reg])[0],
            'Ship Mode Enc': le_s.transform([avg_ship])[0],
            'Product Enc'  : le_p.transform([product])[0],
            'Division Enc' : le_d.transform([division])[0],
            'Sales Sc'     : scaled[0],
            'Cost Sc'      : scaled[1],
            'Units Sc'     : scaled[2],
        }])
        pred = model.predict(X)[0]
        results.append({
            'Factory': fac,
            'Predicted Lead Time': round(pred, 1),
            'Current': 'Current' if fac == curr_fac else 'Alternative'
        })
    res_df = pd.DataFrame(results).sort_values('Predicted Lead Time')
    fig = px.bar(res_df, x='Predicted Lead Time', y='Factory',
                 color='Current', orientation='h',
                 color_discrete_map={'Current': '#E24B4A', 'Alternative': '#178BCA'},
                 title=f"Predicted Lead Time by Factory — {product}")
    st.plotly_chart(fig, use_container_width=True)
    best = res_df.iloc[0]
    if best['Factory'] != curr_fac:
        curr_lt = res_df[res_df['Factory'] == curr_fac]['Predicted Lead Time'].values[0]
        saving  = round(curr_lt - best['Predicted Lead Time'], 1)
        st.success(f"✅ Best factory for {product}: **{best['Factory']}** ({best['Predicted Lead Time']} days) — saves {saving} days")
    else:
        st.info(f"ℹ️ {product} is already in its optimal factory!")

elif page == "🔄 What-If Simulator":
    st.title("🔄 What-If Simulator")
    st.markdown("Compare current vs alternative factory for any product, region and ship mode.")
    col1, col2, col3 = st.columns(3)
    product  = col1.selectbox("Product",   sorted(df['Product Name'].unique()))
    region   = col2.selectbox("Region",    sorted(df['Region'].unique()))
    ship     = col3.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
    prod_df  = df[df['Product Name'] == product]
    division = prod_df['Division'].iloc[0]
    curr_fac = prod_df['Factory'].iloc[0]
    new_fac  = st.selectbox("Alternative Factory",
                            [f for f in all_factories if f != curr_fac])
    avg_sale = prod_df['Sales'].mean()
    avg_cost = prod_df['Cost'].mean()
    avg_unit = prod_df['Units'].mean()
    def predict_lt(fac):
        scaled = scaler.transform([[avg_sale, avg_cost, avg_unit]])[0]
        X = pd.DataFrame([{
            'Factory Enc'  : le_f.transform([fac])[0],
            'Region Enc'   : le_r.transform([region])[0],
            'Ship Mode Enc': le_s.transform([ship])[0],
            'Product Enc'  : le_p.transform([product])[0],
            'Division Enc' : le_d.transform([division])[0],
            'Sales Sc': scaled[0], 'Cost Sc': scaled[1], 'Units Sc': scaled[2]
        }])
        return round(model.predict(X)[0], 1)
    curr_lt = predict_lt(curr_fac)
    new_lt  = predict_lt(new_fac)
    saving  = round(curr_lt - new_lt, 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Lead Time", f"{curr_lt} days")
    c2.metric("New Lead Time",     f"{new_lt} days")
    c3.metric("Days Saved", f"{saving} days", delta=f"{saving}")
    fig2 = px.bar(
        pd.DataFrame({'Factory': [curr_fac, new_fac], 'Lead Time': [curr_lt, new_lt]}),
        x='Factory', y='Lead Time',
        color='Factory', color_discrete_sequence=['#E24B4A', '#51CF66'],
        title="Current vs Alternative Factory Lead Time"
    )
    fig2.update_layout(yaxis_range=[170, 185])
    st.plotly_chart(fig2, use_container_width=True)

elif page == "⭐ Recommendations":
    st.title("⭐ Recommendation Dashboard")
    st.markdown("Ranked factory reassignment recommendations — sorted by total score.")
    priority = st.slider("Optimization Priority — Speed vs Profit (50 = balanced)", 0, 100, 50)
    speed_w  = priority / 100
    profit_w = 1 - speed_w
    disp = recs.copy()
    disp['Weighted Score'] = (disp['LT Score'] * speed_w +
                              disp['Margin Score'] * profit_w).round(1)
    disp = disp.sort_values('Weighted Score', ascending=False).reset_index(drop=True)
    st.dataframe(
        disp[['Product', 'Current Factory', 'New Factory',
              'LT Saving (days)', 'Current Margin %', 'Risk', 'Weighted Score']]
        .head(15),
        use_container_width=True
    )
    fig3 = px.bar(disp.head(10), x='LT Saving (days)', y='Product',
                  color='Risk', orientation='h',
                  color_discrete_map={
                      '🟢 LOW RISK': '#51CF66',
                      '🟡 MEDIUM':   '#f9c74f',
                      '🔴 HIGH RISK':'#E24B4A'
                  },
                  title="Top 10 Recommendations by Lead Time Saving")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "⚠️ Risk Panel":
    st.title("⚠️ Risk & Impact Panel")
    st.markdown("Traffic light view — green = safe to reassign, red = proceed with caution.")
    low    = recs[recs['Risk'] == '🟢 LOW RISK']
    medium = recs[recs['Risk'] == '🟡 MEDIUM']
    high   = recs[recs['Risk'] == '🔴 HIGH RISK']
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Low Risk",    len(low),    "Safe to act")
    c2.metric("🟡 Medium Risk", len(medium), "Review first")
    c3.metric("🔴 High Risk",   len(high),   "Caution")
    st.subheader("🟢 Low Risk Reassignments — Safe to Proceed")
    st.dataframe(low[['Product', 'Current Factory', 'New Factory',
                       'LT Saving (days)', 'Total Score']],
                 use_container_width=True)
    st.subheader("🟡 Medium Risk — Review Before Acting")
    st.dataframe(medium[['Product', 'Current Factory', 'New Factory',
                          'LT Saving (days)', 'Current Margin %']],
                 use_container_width=True)
    if len(high) > 0:
        st.subheader("🔴 High Risk — Do Not Proceed Without Further Analysis")
        st.dataframe(high[['Product', 'Current Factory', 'New Factory',
                            'Current Margin %']], use_container_width=True)
