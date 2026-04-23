import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# --- Load ---
df = pd.read_csv("Lumen_engine_data.csv")
print(f"Loaded {len(df):,} samples across {df['Design_Name'].nunique()} chips\n")

# --- One-Hot Encode recipe steps ---
step_cols = [f'Step_{i}' for i in range(1, 21)]
for col in step_cols:
    df[col] = df[col].astype(str)
df_encoded = pd.get_dummies(df, columns=step_cols)
print(f"One-Hot Encoded: 20 steps -> {df_encoded.shape[1] - 2} features")

# --- 80/20 stratified split per chip ---
train_frames, test_frames = [], []
for chip in df['Design_Name'].unique():
    tr, te = train_test_split(df_encoded[df_encoded['Design_Name'] == chip],
                              test_size=0.2, random_state=42)
    train_frames.append(tr)
    test_frames.append(te)

train_df = pd.concat(train_frames)
test_df  = pd.concat(test_frames)
X_train  = train_df.drop(columns=['Design_Name', 'Delay_Y'])
y_train  = train_df['Delay_Y']
print(f"Split: {len(train_df):,} train / {len(test_df):,} test")

# --- Train ---
model = LinearRegression().fit(X_train, y_train)
print("Model trained.\n")

# --- Per-chip evaluation ---
NAMES = {
    'Rocket_csr':'ROCKET CSR','usb_phy':'USB PHY','jpeg':'JPEG','ss_pcm':'SS PCM',
    'fpu':'FPU','wb_dma':'WB DMA','aes_secworks':'AES SECWORKS',
    'picorv32_typical':'PICORV32 TYP','sasc':'SASC','picorv32_large':'PICORV32 LRG',
    'ac97_ctrl':'AC97 CTRL','aes_ncsu':'AES NCSU','pci':'PCI','or1200':'OR1200',
    'des3_area':'DES3 AREA','Rocket_full':'ROCKET FULL','dynamic_node':'DYNAMIC NODE',
    'mem_ctrl':'MEM CTRL','s35932':'S35932','i2c':'I2C','aes_xcrypt':'AES XCRYPT',
    'or1200_multmac':'OR1200 MULT','picosoc':'PICOSOC','fir':'FIR','spi':'SPI',
    'simple_spi':'SIMPLE SPI','iir':'IIR','aes':'AES','or1200_cpu':'OR1200 CPU',
    'picorv32_small':'PICORV32 SML','Rocket_muldiv':'ROCKET MULD','vga_lcd':'VGA LCD',
    'tv80':'TV80','sha256':'SHA256','bp_be':'BP BE','tinyRocket':'TINYROCKET',
    'or1200_fpu':'OR1200 FPU','wb_conmax':'WB CONMAX','ethernet':'ETHERNET',
    's38417':'S38417','xbar':'XBAR','s38584':'S38584',
}

header = f"{'Target Chip':<18} {'Spearman':>9} {'RMSE(ps)':>12} {'Acc5%':>6} {'Acc10%':>7} {'Gap(ps)':>10}"
print(header)
print("-" * len(header))

for chip in df['Design_Name'].unique():
    te = test_df[test_df['Design_Name'] == chip]
    y_t = te['Delay_Y'].values
    y_p = model.predict(te.drop(columns=['Design_Name', 'Delay_Y']))

    rho, _ = spearmanr(y_t, y_p)
    rmse   = np.sqrt(mean_squared_error(y_t, y_p))
    err    = np.abs((y_p - y_t) / y_t)
    a5, a10 = (err <= 0.05).mean() * 100, (err <= 0.10).mean() * 100

    # gap: predict on ALL data for this chip, pick model's best, compare to true best
    all_c = df_encoded[df_encoded['Design_Name'] == chip]
    y_all = all_c['Delay_Y'].values
    p_all = model.predict(all_c.drop(columns=['Design_Name', 'Delay_Y']))
    gap   = y_all[np.argmin(p_all)] - y_all.min()

    name = NAMES.get(chip, chip.upper())
    print(f"{name:<18} {rho:>+9.4f} {rmse:>12,.2f} {a5:>5.1f}% {a10:>6.1f}% {gap:>+10,.2f}")

# --- Actual vs Predicted plot ---
X_te = test_df.drop(columns=['Design_Name', 'Delay_Y'])
y_te = test_df['Delay_Y'].values
y_pe = model.predict(X_te)
r2   = r2_score(y_te, y_pe)

plt.figure(figsize=(8, 6))
plt.scatter(y_te, y_pe, alpha=0.25, s=12, color='blue', edgecolors='none')
lims = [min(y_te.min(), y_pe.min()), max(y_te.max(), y_pe.max())]
plt.plot(lims, lims, 'k--', lw=1.5, label='Perfect Accuracy')
plt.title(f'Delay: Actual vs Predicted ($R^2={r2:.2f}$)')
plt.xlabel('Actual Delay (ps)')
plt.ylabel('Predicted Delay (ps)')
plt.legend(loc='upper left')
plt.grid(True, ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig("Delay_Actual_vs_Predicted.png", dpi=300)
plt.show()
print("\nPlot saved: Delay_Actual_vs_Predicted.png")
