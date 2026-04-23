import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# 1. Load & Clean the AES Chip Data
# ==========================================
csv_file_path = "/cell_properties.csv"
df = pd.read_csv(csv_file_path)

# Filter out the dead silicon
df_active = df[df['is_filler'] == 0].copy()

# The 5 Layout Features
features = ['x0', 'y0', 'x1', 'y1', 'is_buf']
X = df_active[features]

# The Targets
y_static = df_active['cell_static_power']
y_dynamic = df_active['cell_dynamic_power'].abs()

# Split 80% Training / 20% Unseen Test
X_train, X_test, y_train_static, y_test_static, y_train_dynamic, y_test_dynamic = train_test_split(
    X, y_static, y_dynamic, test_size=0.2, random_state=42
)

# Scale the features so we can compare Beta weights fairly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. Train the Models
# ==========================================
model_static = LinearRegression()
model_static.fit(X_train_scaled, y_train_static)
static_r2 = r2_score(y_test_static, model_static.predict(X_test_scaled))

model_dynamic = LinearRegression()
model_dynamic.fit(X_train_scaled, y_train_dynamic)
dynamic_r2 = r2_score(y_test_dynamic, model_dynamic.predict(X_test_scaled))

# ==========================================
# 3. Print the Equations & Feature Importance
# ==========================================
def print_equation_and_importance(model, target_name, r2):
    print(f"[{target_name.upper()} PREDICTION MODEL]")
    print(f"Unseen Data Accuracy (R^2): {r2:.4f}\n")

    print("Equation:")
    equation = f"{target_name} =\n"
    for feat, coef in zip(features, model.coef_):
        equation += f"  + ({coef:+.4e} * {feat})\n"
    equation += f"  + ({model.intercept_:+.4e} [Intercept])\n"
    print(equation)

    # Calculate Feature Importance (Absolute value of scaled Beta)
    importances = np.abs(model.coef_)
    ranked_indices = np.argsort(importances)[::-1]

    print("Most Influential Features (Ranked):")
    for i, idx in enumerate(ranked_indices):
        print(f"  {i+1}. {features[idx]:<6} (Weight Magnitude: {importances[idx]:.4e})")
    print("-" * 50 + "\n")

print_equation_and_importance(model_static, "Static_Power", static_r2)
print_equation_and_importance(model_dynamic, "Dynamic_Power", dynamic_r2)

# ==========================================
# 4. Generate the Plots
# ==========================================
static_preds = model_static.predict(X_test_scaled)
dynamic_preds = model_dynamic.predict(X_test_scaled)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Static Power
axs[0].scatter(y_test_static, np.abs(static_preds), color='red', alpha=0.3, s=15)
axs[0].plot([y_test_static.min(), y_test_static.max()],
            [y_test_static.min(), y_test_static.max()], 'k--', lw=2, label="Perfect Accuracy")
axs[0].set_title(f'Static Power: Actual vs Predicted (R²={static_r2:.2f})')
axs[0].set_xlabel('Actual Static Power (W)')
axs[0].set_ylabel('Predicted Static Power (W)')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid(True, ls="--", alpha=0.5)
axs[0].legend()

# Plot 2: Dynamic Power
axs[1].scatter(y_test_dynamic, np.abs(dynamic_preds), color='blue', alpha=0.3, s=15)
axs[1].plot([y_test_dynamic.min(), y_test_dynamic.max()],
            [y_test_dynamic.min(), y_test_dynamic.max()], 'k--', lw=2, label="Perfect Accuracy")
axs[1].set_title(f'Dynamic Power: Actual vs Predicted (R²={dynamic_r2:.2f})')
axs[1].set_xlabel('Actual Dynamic Power (W)')
axs[1].set_ylabel('Predicted Dynamic Power (W)')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid(True, ls="--", alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()
