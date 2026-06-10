import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('nassau_clean.csv')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
fig, ax = plt.subplots(figsize=(9, 5))
factory_lt = df.groupby('Factory')['Lead Time'].mean().sort_values(ascending=False)
colors = ['#E24B4A' if v == factory_lt.max() else '#178BCA' for v in factory_lt]
bars = ax.barh(factory_lt.index, factory_lt.values, color=colors)
ax.bar_label(bars, fmt='%.1f days', padding=4)
ax.set_xlabel('Average Lead Time (days)')
ax.set_title('Average Lead Time by Factory', fontweight='bold')
plt.tight_layout()
plt.savefig('chart1_leadtime_factory.png')
plt.close()
fig, ax = plt.subplots(figsize=(9, 5))
order_counts = df['Factory'].value_counts()
colors2 = sns.color_palette('Blues_d', len(order_counts))
bars2 = ax.barh(order_counts.index, order_counts.values, color=colors2)
ax.bar_label(bars2, fmt='%d orders', padding=4)
ax.set_xlabel('Number of Orders')
ax.set_title('Total Orders per Factory', fontweight='bold')
plt.tight_layout()
plt.savefig('chart2_orders_factory.png')
plt.close()
fig, ax = plt.subplots(figsize=(8, 5))
ship_lt = df.groupby('Ship Mode')['Lead Time'].mean().sort_values()
colors3 = sns.color_palette('RdYlGn', len(ship_lt))
bars3 = ax.bar(ship_lt.index, ship_lt.values, color=colors3, edgecolor='white')
ax.bar_label(bars3, fmt='%.1f d', padding=3)
ax.set_ylabel('Average Lead Time (days)')
ax.set_title('Average Lead Time by Ship Mode', fontweight='bold')
ax.set_ylim(170, 182)
plt.tight_layout()
plt.savefig('chart3_leadtime_shipmode.png')
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))
div_rev = df.groupby('Division')['Sales'].sum()
colors4 = ['#5C7CFA', '#FF6B6B', '#51CF66']
wedges, texts, autotexts = ax.pie(
    div_rev.values,
    labels=div_rev.index,
    autopct='%1.1f%%',
    colors=colors4,
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts: at.set_fontsize(13)
ax.set_title('Revenue Share by Division', fontweight='bold')
plt.tight_layout()
plt.savefig('chart4_revenue_division.png')
plt.close()
prod_margin = df.groupby('Product Name')['Gross Margin %'].mean().sort_values()
top5    = prod_margin.tail(5)
bottom5 = prod_margin.head(5)
combined = pd.concat([bottom5, top5])
colors5 = ['#E24B4A']*5 + ['#51CF66']*5

fig, ax = plt.subplots(figsize=(10, 6))
bars5 = ax.barh(combined.index, combined.values, color=colors5)
ax.bar_label(bars5, fmt='%.1f%%', padding=4)
ax.axvline(combined.mean(), color='gray', linestyle='--', label='Average')
ax.set_xlabel('Gross Margin %')
ax.set_title('Top 5 vs Bottom 5 Products — Gross Margin %', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('chart5_margin_products.png')
plt.close()
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    df['Lead Time'], df['Gross Profit'],
    c=df['Gross Profit'], cmap='RdYlGn',
    alpha=0.4, s=15
)
plt.colorbar(scatter, ax=ax, label='Gross Profit ($)')
ax.set_xlabel('Lead Time (days)')
ax.set_ylabel('Gross Profit ($)')
ax.set_title('Lead Time vs Gross Profit', fontweight='bold')
plt.tight_layout()
plt.savefig('chart6_scatter_leadtime_profit.png')
plt.close()
pivot = df.pivot_table(
    values='Lead Time', index='Factory',
    columns='Region', aggfunc='mean'
).round(1)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    pivot, annot=True, fmt='.1f',
    cmap='YlOrRd', ax=ax,
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Avg Lead Time (days)'}
)
ax.set_title('Average Lead Time — Factory × Region', fontweight='bold')
ax.set_xlabel('Region')
ax.set_ylabel('Factory')
plt.tight_layout()
plt.savefig('chart7_heatmap_factory_region.png')
plt.close()
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.to_period('M')
monthly = df.groupby('Month').size().reset_index(name='Orders')
monthly['Month'] = monthly['Month'].astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly['Month'], monthly['Orders'],
       color='#178BCA', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(monthly['Month'], monthly['Orders'], alpha=0.15, color='#178BCA')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Orders')
ax.set_title('Monthly Order Volume Trend', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart8_monthly_trend.png')
plt.close()
