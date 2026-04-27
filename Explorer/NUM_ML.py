# ==============================
# CONCRETE STRENGTH ANALYSIS
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_excel("Concrete_Data.xls")

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== STATISTICS =====")
print(df.describe())

# ==============================
# 2. DATA CHECK
# ==============================
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Correlation
corr = df.corr()

# ==============================
# 3. VISUALIZATION
# ==============================

# 1. Histogram (Target)
plt.figure()
plt.hist(df.iloc[:, -1], bins=30)
plt.title("Distribution of Concrete Strength")
plt.xlabel("Strength")
plt.ylabel("Frequency")
plt.show()

# 2. Cement vs Strength
plt.figure()
plt.scatter(df.iloc[:, 0], df.iloc[:, -1]   )
plt.title("Cement vs Strength")
plt.xlabel("Cement")
plt.ylabel("Strength")
plt.show()

# 3. Age vs Strength
plt.figure()
plt.scatter(df.iloc[:, -2], df.iloc[:, -1])
plt.title("Age vs Strength")
plt.xlabel("Age")
plt.ylabel("Strength")
plt.show()

plt.figure()
plt.boxplot(df.iloc[:, :-1])
plt.title("Boxplot of Features")
plt.xticks(range(1, len(df.columns)), df.columns[:-1], rotation=90)
plt.show()

# 4. Correlation Matrix
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

# ==============================
# 4. MACHINE LEARNING MODEL
# ==============================

# Features & Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# ==============================
# 5. EVALUATION
# ==============================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==============================
# 6. ACTUAL VS PREDICTED
# ==============================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("Actual vs Predicted")
plt.show()

# ==============================
# 7. INFERENCE
# ==============================
print("\n===== INFERENCE =====")
print("1. Cement has strong positive impact on strength.")
print("2. Age increases compressive strength significantly.")
print("3. Water negatively impacts strength when excessive.")
print("4. Model shows moderate accuracy (R2 score).")
print("5. Linear Regression captures trend but not complex patterns.")
print("6. Better models: Random Forest / Gradient Boosting.")