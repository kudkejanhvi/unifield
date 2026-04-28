import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

st.set_page_config(layout='wide', page_title='Churn Explorer')
st.title('European Bank — Churn Explorer')

# locate CSV
expected_name = 'European_Bank.csv'
if os.path.exists(expected_name):
    csv_path = expected_name
else:
    candidates = [f for f in os.listdir() if f.lower().startswith('european_bank') and f.lower().endswith('.csv')]
    if candidates:
        csv_path = candidates[0]
    else:
        csvs = glob.glob('*.csv')
        if csvs:
            csv_path = csvs[0]
        else:
            st.error('No CSV dataset found in app folder.')
            st.stop()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # basic coercions
    for c in ['CreditScore','Age','Tenure','Balance','EstimatedSalary']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # create segments
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,45,60,100], labels=['<30','30-45','46-60','60+'])
    if 'CreditScore' in df.columns:
        df['CreditBand'] = pd.cut(df['CreditScore'], bins=[0,500,700,1000], labels=['Low','Medium','High'])
    if 'Tenure' in df.columns:
        df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1,2,7,20], labels=['New','Mid-term','Long-term'])
    if 'Balance' in df.columns:
        df['BalanceSegment'] = pd.cut(df['Balance'], bins=[-1,0,100000,500000], labels=['Zero Balance','Low Balance','High Balance'])
    return df

df = load_data(csv_path)

# KPIs
st.sidebar.header('Filters')
geo_sel = st.sidebar.multiselect('Geography', options=sorted(df['Geography'].dropna().unique()) if 'Geography' in df.columns else [], default=None)
age_sel = st.sidebar.multiselect('AgeGroup', options=sorted(df['AgeGroup'].dropna().unique()) if 'AgeGroup' in df.columns else [], default=None)

q = df.copy()
if geo_sel:
    q = q[q['Geography'].isin(geo_sel)]
if age_sel:
    q = q[q['AgeGroup'].isin(age_sel)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Customers', len(q))
with col2:
    st.metric('Overall Churn %', f"{q['Exited'].mean()*100:.2f}%")
with col3:
    if 'Balance' in q.columns:
        avg_balance = q['Balance'].median()
        st.metric('Avg Balance', f"{avg_balance:.2f}")
    else:
        st.metric('Avg Balance', 'N/A')

st.subheader('Churn by Geography')
if 'Geography' in q.columns:
    geo = q.groupby('Geography')['Exited'].mean()*100
    st.bar_chart(geo)

st.subheader('Churn by Age Group')
if 'AgeGroup' in q.columns:
    ag = q.groupby('AgeGroup', observed=True)['Exited'].mean()*100
    st.bar_chart(ag)

st.subheader('High-balance Churners (90th pct)')
if 'Balance' in df.columns:
    thr = df['Balance'].quantile(0.90)
    st.write(f'90th percentile balance threshold: {thr:.2f}')
    high = df[(df['Balance']>=thr) & (df['Exited']==1)]
    if not high.empty:
        st.dataframe(high[['CustomerId','Geography','Age','Balance','EstimatedSalary']].head(200))
    else:
        st.write('No high-balance churners in this selection.')

st.markdown('---')
st.write('Data preview:')
st.dataframe(df.head())

st.caption(f'Data loaded from: {csv_path}')
