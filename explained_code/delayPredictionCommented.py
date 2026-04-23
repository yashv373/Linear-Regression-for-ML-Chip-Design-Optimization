# pandas helps read and handle CSV table data
import pandas as pd

# numpy helps with math operations like sqrt(), abs(), arrays
import numpy as np

# matplotlib helps us make graphs and plots
import matplotlib.pyplot as plt

# Linear Regression machine learning model
from sklearn.linear_model import LinearRegression

# used to calculate model accuracy scores like RMSE and R²
from sklearn.metrics import mean_squared_error, r2_score

# splits data into training and testing sets
from sklearn.model_selection import train_test_split

# used to calculate Spearman rank correlation
from scipy.stats import spearmanr


# ==========================================
# 1. Load the Dataset
# ==========================================


# read the CSV file
# this file contains chip synthesis recipe data + delay values
df = pd.read_csv("/OpenABC_delayData.csv")


# print total number of samples
# and total number of unique chips

# len(df) = total rows
# nunique() = total different chip names

print(
    f"Loaded {len(df):,} samples across "
    f"{df['Design_Name'].nunique()} chips\n"
)


# ==========================================
# 2. One-Hot Encode Recipe Steps
# ==========================================


# create column names:
# Step_1 to Step_20

# these represent 20 synthesis recipe steps

step_cols = [f'Step_{i}' for i in range(1, 21)]


# loop through all step columns
for col in step_cols:

    # convert numbers to string

    # this is important because:
    # step values are categories, not real numbers

    # example:
    # step "3" is not bigger than step "2"

    df[col] = df[col].astype(str)


# one-hot encoding converts each step value
# into separate binary columns

# example:
# Step_1 = 3

# becomes:
# Step_1_0 = 0
# Step_1_1 = 0
# Step_1_3 = 1

# this prevents wrong numeric assumptions

df_encoded = pd.get_dummies(
    df,
    columns=step_cols
)


# print how many total features were created

# -2 because:
# Design_Name and Delay_Y are not input features

print(
    f"One-Hot Encoded: 20 steps -> "
    f"{df_encoded.shape[1] - 2} features"
)


# ==========================================
# 3. 80/20 Train-Test Split per Chip
# ==========================================


# empty lists to store training data
# and testing data for every chip

train_frames = []
test_frames = []


# loop through every unique chip name
for chip in df['Design_Name'].unique():

    # select rows only for this chip

    # then split:
    # 80% train
    # 20% test

    tr, te = train_test_split(
        df_encoded[
            df_encoded['Design_Name'] == chip
        ],
        test_size=0.2,
        random_state=42
    )

    # save train part
    train_frames.append(tr)

    # save test part
    test_frames.append(te)


# combine all chip train parts into one dataframe
train_df = pd.concat(train_frames)

# combine all chip test parts into one dataframe
test_df = pd.concat(test_frames)


# X_train = input features

# remove:
# Design_Name → just label
# Delay_Y → target output

X_train = train_df.drop(
    columns=['Design_Name', 'Delay_Y']
)


# y_train = actual delay values we want to predict
y_train = train_df['Delay_Y']


# print split result
print(
    f"Split: {len(train_df):,} train / "
    f"{len(test_df):,} test"
)


# ==========================================
# 4. Train Linear Regression Model
# ==========================================


# create model and train it

# fit() learns best beta values

model = LinearRegression().fit(
    X_train,
    y_train
)


# print confirmation
print("Model trained.\n")


# ==========================================
# 5. Chip Name Mapping
# ==========================================


# dictionary to convert raw dataset names
# into cleaner names for output table

NAMES = {
    'Rocket_csr':'ROCKET CSR',
    'usb_phy':'USB PHY',
    'jpeg':'JPEG',
    'ss_pcm':'SS PCM',
    'fpu':'FPU',
    'wb_dma':'WB DMA',
    'aes_secworks':'AES SECWORKS',
    'picorv32_typical':'PICORV32 TYP',
    'sasc':'SASC',
    'picorv32_large':'PICORV32 LRG',
    'ac97_ctrl':'AC97 CTRL',
    'aes_ncsu':'AES NCSU',
    'pci':'PCI',
    'or1200':'OR1200',
    'des3_area':'DES3 AREA',
    'Rocket_full':'ROCKET FULL',
    'dynamic_node':'DYNAMIC NODE',
    'mem_ctrl':'MEM CTRL',
    's35932':'S35932',
    'i2c':'I2C',
    'aes_xcrypt':'AES XCRYPT',
    'or1200_multmac':'OR1200 MULT',
    'picosoc':'PICOSOC',
    'fir':'FIR',
    'spi':'SPI',
    'simple_spi':'SIMPLE SPI',
    'iir':'IIR',
    'aes':'AES',
    'or1200_cpu':'OR1200 CPU',
    'picorv32_small':'PICORV32 SML',
    'Rocket_muldiv':'ROCKET MULD',
    'vga_lcd':'VGA LCD',
    'tv80':'TV80',
    'sha256':'SHA256',
    'bp_be':'BP BE',
    'tinyRocket':'TINYROCKET',
    'or1200_fpu':'OR1200 FPU',
    'wb_conmax':'WB CONMAX',
    'ethernet':'ETHERNET',
    's38417':'S38417',
    'xbar':'XBAR',
    's38584':'S38584',
}


# ==========================================
# 6. Print Table Header
# ==========================================


# create table heading string

header = (
    f"{'Target Chip':<18} "
    f"{'Spearman':>9} "
    f"{'RMSE(ps)':>12} "
    f"{'Acc5%':>6} "
    f"{'Acc10%':>7} "
    f"{'Gap(ps)':>10}"
)

# print header
print(header)

# print separator line
print("-" * len(header))


# ==========================================
# 7. Per-Chip Evaluation
# ==========================================


# loop through every chip
for chip in df['Design_Name'].unique():

    # take only test rows for this chip
    te = test_df[
        test_df['Design_Name'] == chip
    ]


    # actual delay values
    y_t = te['Delay_Y'].values


    # predicted delay values

    # remove non-feature columns first
    y_p = model.predict(
        te.drop(
            columns=['Design_Name', 'Delay_Y']
        )
    )


    # --------------------------------------
    # Spearman Correlation
    # --------------------------------------

    # checks ranking quality

    # tells:
    # does model know which recipe is better?

    rho, _ = spearmanr(y_t, y_p)


    # --------------------------------------
    # RMSE
    # --------------------------------------

    # root mean square error

    # average prediction error in ps

    rmse = np.sqrt(
        mean_squared_error(y_t, y_p)
    )


    # --------------------------------------
    # Accuracy within 5% and 10%
    # --------------------------------------

    # percentage error

    err = np.abs(
        (y_p - y_t) / y_t
    )


    # accuracy within 5%
    a5 = (err <= 0.05).mean() * 100

    # accuracy within 10%
    a10 = (err <= 0.10).mean() * 100


    # --------------------------------------
    # Gap Calculation
    # --------------------------------------

    # use ALL rows for this chip

    # goal:
    # find model's favorite recipe

    all_c = df_encoded[
        df_encoded['Design_Name'] == chip
    ]


    # true delay values for all recipes
    y_all = all_c['Delay_Y'].values


    # predicted delay values for all recipes
    p_all = model.predict(
        all_c.drop(
            columns=['Design_Name', 'Delay_Y']
        )
    )


    # np.argmin(p_all)

    # finds index of smallest predicted delay
    # meaning model's best recipe choice

    # compare that recipe's true delay
    # with actual best delay

    gap = (
        y_all[np.argmin(p_all)]
        - y_all.min()
    )


    # get clean chip name from dictionary

    # if not found,
    # use uppercase raw name

    name = NAMES.get(
        chip,
        chip.upper()
    )


    # print final row for this chip
    print(
        f"{name:<18} "
        f"{rho:>+9.4f} "
        f"{rmse:>12,.2f} "
        f"{a5:>5.1f}% "
        f"{a10:>6.1f}% "
        f"{gap:>+10,.2f}"
    )


# ==========================================
# 8. Actual vs Predicted Plot
# ==========================================


# test input features
X_te = test_df.drop(
    columns=['Design_Name', 'Delay_Y']
)


# real delay values
y_te = test_df['Delay_Y'].values


# predicted delay values
y_pe = model.predict(X_te)


# R² score for full test set
r2 = r2_score(
    y_te,
    y_pe
)


# create figure
plt.figure(figsize=(8, 6))


# scatter plot

# x-axis = actual delay
# y-axis = predicted delay

plt.scatter(
    y_te,
    y_pe,
    alpha=0.25,
    s=12,
    color='blue',
    edgecolors='none'
)


# create limits for perfect line

# line = y = x

lims = [
    min(y_te.min(), y_pe.min()),
    max(y_te.max(), y_pe.max())
]


# draw perfect prediction line

plt.plot(
    lims,
    lims,
    'k--',                    # black dashed line
    lw=1.5,
    label='Perfect Accuracy'
)


# graph title with R²
plt.title(
    f'Delay: Actual vs Predicted ($R^2={r2:.2f}$)'
)


# x-axis label
plt.xlabel('Actual Delay (ps)')


# y-axis label
plt.ylabel('Predicted Delay (ps)')


# show legend
plt.legend(
    loc='upper left'
)


# add background grid
plt.grid(
    True,
    ls=':',
    alpha=0.5
)


# fix spacing
plt.tight_layout()


# save image as PNG

# dpi=300 means high quality
plt.savefig(
    "Delay_Actual_vs_Predicted.png",
    dpi=300
)


# show plot on screen
plt.show()


# print final confirmation
print(
    "\nPlot saved: Delay_Actual_vs_Predicted.png"
)
