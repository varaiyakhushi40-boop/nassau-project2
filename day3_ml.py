import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('nassau_clean.csv')
le = LabelEncoder()
df['Ship Mode Enc']  = le.fit_transform(df['Ship Mode'])
df['Region Enc']     = le.fit_transform(df['Region'])
df['Factory Enc']    = le.fit_transform(df['Factory'])
df['Product Enc']    = le.fit_transform(df['Product Name'])
df['Division Enc']   = le.fit_transform(df['Division'])
scaler = MinMaxScaler()
df[['Sales Sc', 'Cost Sc', 'Units Sc']] = scaler.fit_transform(
    df[['Sales', 'Cost', 'Units']]
)
features = ['Factory Enc', 'Region Enc', 'Ship Mode Enc',
            'Product Enc', 'Division Enc',
            'Sales Sc', 'Cost Sc', 'Units Sc']
X = df[features]
y = df['Lead Time']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training rows :", len(X_train))
print("Testing rows  :", len(X_test))
print("Features used :", features)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
r2_lr   = r2_score(y_test, y_pred_lr)

print("── Linear Regression ──")
print(f"RMSE : {rmse_lr:.3f}")
print(f"MAE  : {mae_lr:.3f}")
print(f"R²   : {r2_lr:.3f}")
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
r2_rf   = r2_score(y_test, y_pred_rf)

print("── Random Forest ──")
print(f"RMSE : {rmse_rf:.3f}")
print(f"MAE  : {mae_rf:.3f}")
print(f"R²   : {r2_rf:.3f}")
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb  = mean_absolute_error(y_test, y_pred_gb)
r2_gb   = r2_score(y_test, y_pred_gb)

print("── Gradient Boosting ──")
print(f"RMSE : {rmse_gb:.3f}")
print(f"MAE  : {mae_gb:.3f}")
print(f"R²   : {r2_gb:.3f}")
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'RMSE':  [round(rmse_lr,3), round(rmse_rf,3), round(rmse_gb,3)],
    'MAE':   [round(mae_lr,3),  round(mae_rf,3),  round(mae_gb,3)],
    'R²':    [round(r2_lr,3),   round(r2_rf,3),   round(r2_gb,3)]
})
print("Model Comparison Table:")
print(results.to_string(index=False))
best_idx   = results['RMSE'].idxmin()
best_name  = results.loc[best_idx, 'Model']
print(f"Best model: {best_name}")
results.to_csv('model_comparison.csv', index=False)
importances = pd.Series(gb.feature_importances_, index=features).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#178BCA' if v < importances.max() else '#E24B4A' for v in importances]
bars = ax.barh(importances.index, importances.values, color=colors)
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance — Gradient Boosting', fontweight='bold')
plt.tight_layout()
plt.savefig('chart9_feature_importance.png')
plt.close()
print("Top feature:", importances.idxmax())
import joblib
joblib.dump(gb, 'best_model.pkl')
joblib.dump(features, 'features.pkl')
joblib.dump(scaler,   'scaler.pkl')
print("Files saved:")
print("  best_model.pkl  — Gradient Boosting model")
print("  features.pkl    — list of feature column names")
print("  scaler.pkl      — MinMaxScaler for new data")