# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, mean_squared_error

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv(r"D:\sem 6\data science\ds practice\data.csv")

# Example dataset format
# Age  Study Hours  Marks

# ---------------------------
# 1. EDA
# ---------------------------
print(df.head())
print(df.describe())

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# ---------------------------
# 2. Data Preprocessing
# ---------------------------

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Study Hours', 'Marks']] = imputer.fit_transform(
    df[['Age', 'Study Hours', 'Marks']]
)

# Features and Target
X = df[['Age', 'Study Hours']]
y = df['Marks']

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 3. Visualization
# ---------------------------
plt.scatter(df['Study Hours'], df['Marks'])
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Study Hours vs Marks')
plt.show()

# ---------------------------
# 4. Classification (PassFail)
# ---------------------------

# Convert Marks → Pass(1)  Fail(0)
y_class = (df['Marks'] >= 50).astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

clf = LogisticRegression()
clf.fit(X_train_c, y_train_c)

y_pred = clf.predict(X_test_c)

print('Classification Accuracy', accuracy_score(y_test_c, y_pred))

# ---------------------------
# 5. Regression (Predict Marks)
# ---------------------------
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

print('Regression MSE', mean_squared_error(y_test, y_pred_reg))

# ---------------------------
# 6. Clustering
# ---------------------------
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X)

plt.scatter(df['Study Hours'], df['Marks'], c=clusters)
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Clustering Students')
plt.show()
