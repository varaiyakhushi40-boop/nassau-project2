import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
df       = pd.read_csv('nassau_clean.csv')
model    = joblib.load('best_model.pkl')
features = joblib.load('features.pkl')
scaler   = joblib.load('scaler.pkl')
all_factories = [
    "Lot's O' Nuts", "Wicked Choccy's",
    'Sugar Shack', 'Secret Factory', 'The Other Factory'
]
from sklearn.preprocessing import LabelEncoder
le_factory = LabelEncoder()
le_factory.fit(df['Factory'])
le_region  = LabelEncoder()
le_region.fit(df['Region'])
le_ship    = LabelEncoder()
le_ship.fit(df['Ship Mode'])
le_product = LabelEncoder()
le_product.fit(df['Product Name'])
le_div     = LabelEncoder()
le_div.fit(df['Division'])
current = df.groupby(['Product Name', 'Factory', 'Division']).agg(
    Current_Lead_Time  = ('Lead Time',     'mean'),
    Current_Margin_Pct = ('Gross Margin %', 'mean'),
    Total_Orders       = ('Lead Time',     'count')
).round(2).reset_index()

print("Current performance (first 5 products):")
print(current[['Product Name','Factory',
               'Current_Lead_Time','Current_Margin_Pct']]
      .head(5).to_string(index=False))
sim_results = []

for _, row in current.iterrows():
    product  = row['Product Name']
    curr_fac = row['Factory']
    curr_lt  = row['Current_Lead_Time']
    curr_mg  = row['Current_Margin_Pct']
prod_df  = df[df['Product Name'] == product]
avg_sale = prod_df['Sales'].mean()
avg_cost = prod_df['Cost'].mean()
avg_unit = prod_df['Units'].mean()
avg_reg  = prod_df['Region'].mode()[0]
avg_ship = prod_df['Ship Mode'].mode()[0]
division = row['Division']

for new_fac in all_factories:
 if new_fac == curr_fac:continue
scaled = scaler.transform([[avg_sale, avg_cost, avg_unit]])[0]
X_sim  = pd.DataFrame([{
            'Factory Enc'   : le_factory.transform([new_fac])[0],
            'Region Enc'    : le_region.transform([avg_reg])[0],
            'Ship Mode Enc' : le_ship.transform([avg_ship])[0],
            'Product Enc'   : le_product.transform([product])[0],
            'Division Enc'  : le_div.transform([division])[0],
            'Sales Sc'      : scaled[0],
            'Cost Sc'       : scaled[1],
            'Units Sc'      : scaled[2],
        }])
pred_lt = model.predict(X_sim)[0]
lt_save = curr_lt - pred_lt
sim_results.append({
            'Product'         : product,
            'Division'        : division,
            'Current Factory' : curr_fac,
            'New Factory'     : new_fac,
            'Current LT'      : round(curr_lt,  1),
            'Predicted LT'    : round(pred_lt, 1),
            'LT Saving (days)': round(lt_save,  1),
            'Current Margin %': round(curr_mg,  1),
        })

sim_df = pd.DataFrame(sim_results)
print(f"Total simulations run: {len(sim_df)}")
sim_df = sim_df[sim_df['LT Saving (days)'] > 0].copy()
max_lt_save = sim_df['LT Saving (days)'].max()
sim_df['LT Score']     = (sim_df['LT Saving (days)'] / max_lt_save * 60).round(1)
sim_df['Margin Score']  = sim_df['Current Margin %'].apply(
    lambda x: 40 if x >= 28 else round(x / 28 * 40, 1)
)
sim_df['Total Score']   = (sim_df['LT Score'] + sim_df['Margin Score']).round(1)
sim_df['Risk'] = sim_df['Current Margin %'].apply(
    lambda x: '🔴 HIGH RISK' if x < 25 else ('🟡 MEDIUM' if x < 30 else '🟢 LOW RISK')
)
sim_df = sim_df.sort_values('Total Score', ascending=False).reset_index(drop=True)

print("Top 5 Recommendations:")
print(sim_df[['Product','Current Factory','New Factory',
'LT Saving (days)','Total Score','Risk']]
.head(5).to_string(index=False))
sim_df.to_csv('recommendations.csv', index=False)
top10 = sim_df.head(10).copy()
top10['Label'] = (
    top10['Product'].str.replace('Wonka Bar - ', 'WB - ')
    + ' - '
    + top10['New Factory']
)
colors = ['#E24B4A' if r=='🔴 HIGH RISK' else '#f9c74f' if r=='🟡 MEDIUM' else '#51CF66'
          for r in top10['Risk']]
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top10['Label'], top10['LT Saving (days)'], color=colors)
ax.bar_label(bars, fmt='%.1f days saved', padding=4)
ax.set_xlabel('Lead Time Saved (days)')
ax.set_title('Top 10 Factory Reassignment Recommendations', fontweight='bold')
from matplotlib.patches import Patch
legend = [Patch(color='#51CF66',label='Low Risk'),
          Patch(color='#f9c74f',label='Medium Risk'),
          Patch(color='#E24B4A',label='High Risk')]
ax.legend(handles=legend, loc='lower right')
plt.tight_layout()
plt.savefig('chart10_recommendations.png')
plt.close()
print("recommendations.csv saved!")
print("chart10_recommendations.png saved!")
top5 = sim_df.head(5)
print("="*60)
print("TOP 5 FACTORY REASSIGNMENT RECOMMENDATIONS")
print("Nassau Candy Distributor — Project 2")
print("="*60)
for i, row in top5.iterrows():
    print(f"Rec #{i+1}: {row['Product']}")
    print(f"  Move from : {row['Current Factory']}")
    print(f"  Move to   : {row['New Factory']}")
    print(f"  Lead time : {row['Current LT']} → {row['Predicted LT']} days")
    print(f"  Saving    : {row['LT Saving (days)']} days per order")
    print(f"  Margin    : {row['Current Margin %']}%  |  Risk: {row['Risk']}")
    print(f"  Score     : {row['Total Score']} / 100")

print("" + "="*60)
