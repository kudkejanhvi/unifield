# CUSTOMER SEGMENTATION & CHURN ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. CHECK WORKING DIRECTORY

print("Current Directory:", os.getcwd())
print("Files Available:", os.listdir())

# 2. LOAD DATA

# Resolve CSV filename robustly (handles extra spaces or different names)
expected_name = "European_Bank.csv"
if os.path.exists(expected_name):
      csv_path = expected_name
else:
      candidates = [f for f in os.listdir() if f.lower().startswith('european_bank') and f.lower().endswith('.csv')]
      if candidates:
            csv_path = candidates[0]
            print("Using detected file:", csv_path)
      else:
            import glob
            csvs = glob.glob("*.csv")
            if len(csvs) == 1:
                  csv_path = csvs[0]
                  print("Using only CSV present:", csv_path)
            elif csvs:
                  print("Multiple CSV files found:", csvs)
                  raise FileNotFoundError(f"Expected '{expected_name}' but found: {csvs}")
            else:
                  raise FileNotFoundError(f"No CSV file found in {os.getcwd()}")

df = pd.read_csv(csv_path)

# Data Ingestion & Validation

required_cols = ['CustomerId','Surname','CreditScore','Geography','Gender','Age',
                         'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember',
                         'EstimatedSalary','Exited']
missing = [c for c in required_cols if c not in df.columns]
if missing:
      print("Warning - missing expected columns:", missing)

# Validate engagement and product fields
for col in ['IsActiveMember','HasCrCard']:
      if col in df.columns:
            vals = sorted(df[col].dropna().unique())
            print(f"{col} unique values:", vals)
            if not all(v in (0,1) for v in vals):
                  print(f"Note: {col} contains non-binary values")

# Confirm churn labeling
if 'Exited' in df.columns:
      print("Exited unique:", sorted(df['Exited'].dropna().unique()))
      if not all(v in (0,1) for v in df['Exited'].dropna().unique()):
            print("Warning: 'Exited' contains non-binary labels")

# 3. BASIC VALIDATION

print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nChurn Distribution:\n", df['Exited'].value_counts())


# 4. DATA CLEANING

# Keep `CustomerId` for traceability/reporting; drop only `Surname`
if 'Surname' in df.columns:
      df.drop(['Surname'], axis=1, inplace=True)

df['Gender'] = df['Gender'].astype('category')
df['Geography'] = df['Geography'].astype('category')

# Ensure numeric columns are numeric
num_cols = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']
for c in num_cols:
      if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')


# 5. FEATURE ENGINEERING


# Age Group
df['AgeGroup'] = pd.cut(df['Age'],
                        bins=[18,30,45,60,100],
                        labels=['<30','30-45','46-60','60+'])

# Credit Band
df['CreditBand'] = pd.cut(df['CreditScore'],
                          bins=[0,500,700,1000],
                          labels=['Low','Medium','High'])

# Tenure Group
df['TenureGroup'] = pd.cut(df['Tenure'],
                           bins=[-1,2,7,20],
                           labels=['New','Mid-term','Long-term'])

# Balance Segment
df['BalanceSegment'] = pd.cut(df['Balance'],
                              bins=[-1,0,100000,500000],
                              labels=['Zero Balance','Low Balance','High Balance'])


# Churn Distribution Analysis


overall_churn = df['Exited'].mean()*100
print("\nOverall Churn Rate:", round(overall_churn,2), "%")

def segment_churn(series):
      return (df.groupby(series)['Exited'].mean()*100).sort_values(ascending=False)

print("\nGeography-wise Churn:\n", segment_churn('Geography'))
print("\nAge-wise Churn:\n", segment_churn('AgeGroup'))
print("\nCredit-band Churn:\n", segment_churn('CreditBand'))
print("\nTenure-wise Churn:\n", segment_churn('TenureGroup'))
print("\nBalance Segment Churn:\n", segment_churn('BalanceSegment'))

# Churn contribution by segment size (example: geography)
geo = df.groupby('Geography').agg(total=('Exited','size'), churned=('Exited','sum'))
geo['churn_rate'] = geo['churned']/geo['total']*100
geo['churn_contribution_pct'] = geo['churned']/geo['churned'].sum()*100
print("\nGeography churn contribution:\n", geo.sort_values('churn_contribution_pct', ascending=False))

# Compare churned vs retained profiles
profile_cols = ['Age','CreditScore','Balance','EstimatedSalary','NumOfProducts','Tenure']
existing = [c for c in profile_cols if c in df.columns]
print("\nChurned vs Retained (means):\n", df.groupby('Exited')[existing].mean())


# Comparative Demographic Analysis


if 'Gender' in df.columns:
      print("\nGender-wise churn (%%):\n", df.groupby('Gender')['Exited'].mean()*100)

# Geography-age interaction (pivot)
if 'AgeGroup' in df.columns and 'Geography' in df.columns:
      pivot = pd.pivot_table(df, index='AgeGroup', columns='Geography', values='Exited', aggfunc='mean')*100
      print("\nGeography x AgeGroup churn (%):\n", pivot)

# Financial stability vs churn (Balance & Salary)
if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
      bal_salary = df.groupby('Exited')[['Balance','EstimatedSalary']].median()
      print("\nMedian Balance & Salary by churn status:\n", bal_salary)


# High-Value Customer Churn Analysis


# Identify high-balance churners
high_balance_thr = df['Balance'].quantile(0.90) if 'Balance' in df.columns else None
if high_balance_thr is not None:
      high_churners = df[(df['Balance']>=high_balance_thr) & (df['Exited']==1)]
      print(f"\nHigh-balance threshold (90th pct): {round(high_balance_thr,2)}")
      print("High-balance churners count:", len(high_churners))
      if len(high_churners):
            print(high_churners[['CustomerId','Geography','Age','Balance','EstimatedSalary']].head())

# Compare salary vs balance churn patterns
if 'EstimatedSalary' in df.columns and 'Balance' in df.columns:
      agg = df.groupby('Exited')[['Balance','EstimatedSalary']].agg(['mean','median'])
      print("\nBalance & Salary (mean/median) by churn status:\n", agg)

# Quantify revenue risk from churn (sum of balances of churned customers)
if 'Balance' in df.columns:
      at_risk_balance = df[df['Exited']==1]['Balance'].sum()
      total_balance = df['Balance'].sum()
      print(f"\nTotal balance at risk due to churn: {at_risk_balance:.2f} (of {total_balance:.2f}, {at_risk_balance/total_balance*100 if total_balance>0 else 0:.2f}%)")


# Export summary tables for reporting

out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)
try:
      geo.to_csv(os.path.join(out_dir, 'geography_churn_contribution.csv'))
except Exception:
      pass
try:
      df.groupby('Exited')[existing].mean().to_csv(os.path.join(out_dir, 'churned_vs_retained_means.csv'))
except Exception:
      pass
try:
      if 'AgeGroup' in df.columns and 'Geography' in df.columns:
            pivot.to_csv(os.path.join(out_dir, 'geography_age_churn_pivot.csv'))
except Exception:
      pass
try:
      if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
            bal_salary.to_csv(os.path.join(out_dir, 'median_balance_salary_by_churn.csv'))
except Exception:
      pass
try:
      if high_balance_thr is not None and len(high_churners):
            high_churners.to_csv(os.path.join(out_dir, 'high_balance_churners.csv'), index=False)
except Exception:
      pass

print(f"\nExported summary tables (if available) to: {out_dir}")


# Visualizations (basic)


try:
      plt.figure(figsize=(7,4))
      (df.groupby('Geography')['Exited'].mean()*100).plot(kind='bar')
      plt.title("Geography-wise Churn Rate")
      plt.ylabel("Churn %")

      plt.figure(figsize=(7,4))
      (df.groupby('AgeGroup')['Exited'].mean()*100).plot(kind='bar')
      plt.title("Age-wise Churn Rate")
      plt.ylabel("Churn %")

      plt.figure(figsize=(7,4))
      (df.groupby('CreditBand')['Exited'].mean()*100).plot(kind='bar')
      plt.title("CreditBand Churn Rate")
      plt.ylabel("Churn %")

      # Scatter salary vs balance colored by churn
      if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
            plt.figure(figsize=(6,5))
            colors = df['Exited'].map({0:'C0',1:'C1'})
            plt.scatter(df['EstimatedSalary'], df['Balance'], c=colors, alpha=0.6)
            plt.xlabel('EstimatedSalary')
            plt.ylabel('Balance')
            plt.title('Salary vs Balance (colored by Exited)')

      plt.show()
except Exception as e:
      print('Plotting skipped due to:', e)


# 6. KPI CALCULATIONS


overall_churn = df['Exited'].mean()*100
print("\nOverall Churn Rate:", round(overall_churn,2), "%")

print("\nGeography-wise Churn:\n",
      df.groupby('Geography')['Exited'].mean()*100)

print("\nAge-wise Churn:\n",
      df.groupby('AgeGroup')['Exited'].mean()*100)

print("\nGender-wise Churn:\n",
      df.groupby('Gender')['Exited'].mean()*100)

inactive_churn = df[df['IsActiveMember']==0]['Exited'].mean()*100
active_churn = df[df['IsActiveMember']==1]['Exited'].mean()*100

print("\nInactive Churn:", round(inactive_churn,2))
print("Active Churn:", round(active_churn,2))


# 7. VISUALIZATION


plt.figure()
(df.groupby('Geography')['Exited'].mean()*100).plot(kind='bar')
plt.title("Geography-wise Churn Rate")
plt.ylabel("Churn %")
plt.show()

plt.figure()
(df.groupby('AgeGroup')['Exited'].mean()*100).plot(kind='bar')
plt.title("Age-wise Churn Rate")
plt.ylabel("Churn %")
plt.show()
