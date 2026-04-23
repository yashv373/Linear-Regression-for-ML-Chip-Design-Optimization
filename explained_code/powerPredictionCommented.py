# pandas helps us read and handle CSV table data
import pandas as pd

# numpy helps with math operations like abs(), sorting, arrays
import numpy as np

# matplotlib helps us make graphs and plots
import matplotlib.pyplot as plt

# splits data into training data and testing data
from sklearn.model_selection import train_test_split

# scales feature values so all features are treated fairly
from sklearn.preprocessing import StandardScaler

# Linear Regression machine learning model
from sklearn.linear_model import LinearRegression

# checks model accuracy using R² score
from sklearn.metrics import r2_score


# ==========================================
# 1. Load and Clean the AES Chip Data
# ==========================================

# path of the CSV file
csv_file_path = "/cell_properties.csv"

# read CSV file and store it in a dataframe called df
df = pd.read_csv(csv_file_path)


# remove filler cells
# filler cells are dummy silicon and have almost zero power
# keep only real active cells where is_filler = 0
df_active = df[df['is_filler'] == 0].copy()


# these are the input features for prediction

# x0 = bottom-left x coordinate
# y0 = bottom-left y coordinate
# x1 = top-right x coordinate
# y1 = top-right y coordinate
# is_buf = tells if the cell is a buffer or not

features = ['x0', 'y0', 'x1', 'y1', 'is_buf']

# take only these columns from dataframe
# X = input features
X = df_active[features]


# target 1 = static power (leakage power)
y_static = df_active['cell_static_power']

# target 2 = dynamic power (switching power)

# abs() makes all values positive
# because power values should be positive
y_dynamic = df_active['cell_dynamic_power'].abs()


# split data into:
# 80% for training
# 20% for testing

# random_state=42 means same split every time

X_train, X_test, y_train_static, y_test_static, y_train_dynamic, y_test_dynamic = train_test_split(
    X,                  # input features
    y_static,           # static power target
    y_dynamic,          # dynamic power target
    test_size=0.2,      # 20% test data
    random_state=42     # fixed random split
)


# create scaler object

# this is needed because:
# x coordinates may be huge values
# is_buf is only 0 or 1

# scaling makes feature comparison fair
scaler = StandardScaler()


# learn scaling using training data
# and apply scaling on training data
X_train_scaled = scaler.fit_transform(X_train)


# apply same scaling to test data

# important:
# do not fit again on test data
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 2. Train the Models
# ==========================================


# create Linear Regression model for static power
model_static = LinearRegression()


# train the model using:
# scaled training features + static power output
model_static.fit(X_train_scaled, y_train_static)


# predict static power on test data
# then calculate R² accuracy score

# R² closer to 1 means better model
static_r2 = r2_score(
    y_test_static,                      # real values
    model_static.predict(X_test_scaled) # predicted values
)


# create another Linear Regression model
# this one is for dynamic power
model_dynamic = LinearRegression()


# train dynamic power model
model_dynamic.fit(X_train_scaled, y_train_dynamic)


# predict dynamic power and calculate R²
dynamic_r2 = r2_score(
    y_test_dynamic,                     # real values
    model_dynamic.predict(X_test_scaled) # predicted values
)


# ==========================================
# 3. Print Equation and Feature Importance
# ==========================================


# function to print:
# 1. model accuracy
# 2. regression equation
# 3. feature importance ranking

def print_equation_and_importance(model, target_name, r2):

    # print model heading
    print(f"[{target_name.upper()} PREDICTION MODEL]")


    # print R² score with 4 decimal places
    print(f"Unseen Data Accuracy (R^2): {r2:.4f}\n")


    # print heading
    print("Equation:")


    # start building equation text
    equation = f"{target_name} =\n"


    # loop through every feature and its beta coefficient

    # zip() joins:
    # feature names + model coefficients

    for feat, coef in zip(features, model.coef_):

        # add each term like:
        # + (beta × feature)

        equation += f"  + ({coef:+.4e} * {feat})\n"


    # add intercept term

    # intercept = constant base value
    equation += f"  + ({model.intercept_:+.4e} [Intercept])\n"


    # print full equation
    print(equation)


    # Feature Importance

    # take absolute value of coefficients

    # because:
    # big negative is also important
    # big positive is also important

    importances = np.abs(model.coef_)


    # sort feature importance from highest to lowest

    # argsort() gives index positions
    # [::-1] reverses order to descending

    ranked_indices = np.argsort(importances)[::-1]


    # print heading
    print("Most Influential Features (Ranked):")


    # loop through sorted feature indexes
    for i, idx in enumerate(ranked_indices):

        # print:
        # rank number
        # feature name
        # weight size

        print(
            f"  {i+1}. {features[idx]:<6} "
            f"(Weight Magnitude: {importances[idx]:.4e})"
        )


    # print separator line
    print("-" * 50 + "\n")


# run function for static power model
print_equation_and_importance(
    model_static,
    "Static_Power",
    static_r2
)


# run function for dynamic power model
print_equation_and_importance(
    model_dynamic,
    "Dynamic_Power",
    dynamic_r2
)


# ==========================================
# 4. Generate the Plots
# ==========================================


# predict static power values using test data
static_preds = model_static.predict(X_test_scaled)


# predict dynamic power values using test data
dynamic_preds = model_dynamic.predict(X_test_scaled)


# create 2 plots side by side

# 1 row
# 2 columns

# figsize = width 14, height 6
fig, axs = plt.subplots(1, 2, figsize=(14, 6))


# ==========================================
# Plot 1 : Static Power
# ==========================================


# scatter plot

# x-axis = actual static power
# y-axis = predicted static power

# red points
# alpha = transparency
# s = point size

axs[0].scatter(
    y_test_static,
    np.abs(static_preds),
    color='red',
    alpha=0.3,
    s=15
)


# draw perfect prediction line

# this is y = x line

# if points lie on this line
# prediction is perfect

axs[0].plot(
    [y_test_static.min(), y_test_static.max()],
    [y_test_static.min(), y_test_static.max()],
    'k--',                     # black dashed line
    lw=2,                      # line width
    label="Perfect Accuracy"   # legend name
)


# graph title with R² score
axs[0].set_title(
    f'Static Power: Actual vs Predicted (R²={static_r2:.2f})'
)


# x-axis label
axs[0].set_xlabel('Actual Static Power (W)')


# y-axis label
axs[0].set_ylabel('Predicted Static Power (W)')


# use log scale because values vary a lot
axs[0].set_xscale('log')
axs[0].set_yscale('log')


# add background grid
axs[0].grid(
    True,
    ls="--",   # dashed lines
    alpha=0.5  # transparency
)


# show legend box
axs[0].legend()


# ==========================================
# Plot 2 : Dynamic Power
# ==========================================


# same idea for dynamic power plot

axs[1].scatter(
    y_test_dynamic,
    np.abs(dynamic_preds),
    color='blue',
    alpha=0.3,
    s=15
)


# perfect prediction line
axs[1].plot(
    [y_test_dynamic.min(), y_test_dynamic.max()],
    [y_test_dynamic.min(), y_test_dynamic.max()],
    'k--',
    lw=2,
    label="Perfect Accuracy"
)


# graph title with R² score
axs[1].set_title(
    f'Dynamic Power: Actual vs Predicted (R²={dynamic_r2:.2f})'
)


# x-axis label
axs[1].set_xlabel('Actual Dynamic Power (W)')


# y-axis label
axs[1].set_ylabel('Predicted Dynamic Power (W)')


# log scale again
axs[1].set_xscale('log')
axs[1].set_yscale('log')


# background grid
axs[1].grid(
    True,
    ls="--",
    alpha=0.5
)


# show legend
axs[1].legend()


# automatically fix spacing between plots
plt.tight_layout()


# display final plots
plt.show()
