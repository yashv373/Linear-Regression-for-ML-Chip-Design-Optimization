# Linear Regression for ML-Aided Chip Design Optimization
---
colab view link: https://colab.research.google.com/drive/1tPIh_t2jS-l1nhfsIl9iItkd0qMTUR7A?usp=sharing
---
This project explores the use of **Linear Regression as a baseline machine learning model in VLSI design automation (EDA)**. Instead of directly relying on heavy black-box models like Graph Neural Networks, this work studies where simple linear models are sufficient and where they fail.

The project covers two case studies:

1. **Synthesis Delay Prediction** using the OpenABC-D dataset, where synthesis recipes and chip structural features are used to predict post-synthesis delay.

2. **Spatial Power Prediction** using the NVIDIA CircuitOps dataset, where floorplan coordinates (`x0, y0, x1, y1`) and structural cell features are used to predict static and dynamic power dissipation in an AES core on sky130hd.

The results reveal a clear **“Linearity Gap”**:
Linear Regression performs well for structurally linear problems like static leakage estimation, but struggles with highly non-linear problems like dynamic power and complex synthesis timing prediction.

This work shows that Linear Regression should not be treated as just a weak baseline, but as a fast, interpretable, and practical optimization tool for specific stages of the chip design flow. 
