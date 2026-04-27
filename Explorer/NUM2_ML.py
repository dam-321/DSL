# ==========================================
# CONCRETE STRENGTH ANALYSIS (ADVANCED)
# EDA + VISUALIZATION + ML MODELS + REASONING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_excel("Concrete_Data.xls")

print("===== DATA PREVIEW =====")
print(df.head())

print("\n===== STATISTICS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ==========================================
# 2. CORRELATION
# ==========================================
corr = df.corr()

# ==========================================
# 3. VISUALIZATION (4 PLOTS)
# ==========================================

# Histogram
plt.figure()
plt.hist(df.iloc[:, -1], bins=30)
plt.title("Strength Distribution")
plt.xlabel("Strength")
plt.ylabel("Frequency")
plt.show()

# Cement vs Strength
plt.figure()
plt.scatter(df.iloc[:, 0], df.iloc[:, -1])
plt.title("Cement vs Strength")
plt.xlabel("Cement")
plt.ylabel("Strength")
plt.show()

# Age vs Strength
plt.figure()
plt.scatter(df.iloc[:, -2], df.iloc[:, -1])
plt.title("Age vs Strength")
plt.xlabel("Age")
plt.ylabel("Strength")
plt.show()

# Correlation Heatmap
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 4. DATA PREPARATION
# ==========================================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling (IMPORTANT for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. MODELS
# ==========================================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# SVM (SVR)
svm = SVR(kernel='rbf')
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================
def evaluate(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n===== {name} PERFORMANCE =====")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# Evaluate all
evaluate("Linear Regression", y_test, lr_pred)
evaluate("Decision Tree", y_test, dt_pred)
evaluate("SVM", y_test, svm_pred)

# ==========================================
# 7. VISUAL COMPARISON
# ==========================================
plt.figure()
plt.scatter(y_test, lr_pred)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

plt.figure()
plt.scatter(y_test, dt_pred)
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

plt.figure()
plt.scatter(y_test, svm_pred)
plt.title("SVM: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# ==========================================
# 8. REASON-BASED INFERENCE (VERY IMPORTANT)
# ==========================================
print("\n===== DETAILED INFERENCE (WITH REASONS) =====\n")

print("1. WHY Cement increases strength:")
print("   Cement undergoes hydration reaction with water, forming calcium silicate hydrate (C-S-H gel).")
print("   This gel is responsible for binding aggregates together, increasing structural strength.")
print("   Hence, higher cement content → more bonding material → higher compressive strength.\n")

print("2. WHY Age increases strength:")
print("   Concrete gains strength over time due to continuous hydration.")
print("   The chemical reaction between cement and water is slow and continues for days/weeks.")
print("   Longer curing → more complete reaction → stronger internal structure.\n")

print("3. WHY Water reduces strength (after limit):")
print("   Water is necessary for hydration, but excess water creates pores after evaporation.")
print("   These pores weaken the internal structure of concrete.")
print("   So, high water content → higher porosity → lower strength.\n")

print("4. WHY Linear Regression performs moderate:")
print("   Linear Regression assumes a linear relationship between input and output.")
print("   However, concrete strength depends on complex chemical interactions (non-linear).")
print("   Hence, it cannot fully capture the real-world behavior.\n")

print("5. WHY Decision Tree may overfit:")
print("   Decision Trees split data into many small regions.")
print("   This allows capturing non-linear patterns but may memorize training data.")
print("   Result: High training accuracy but unstable predictions on test data.\n")

print("6. WHY SVM performs differently:")
print("   SVM (with RBF kernel) maps data into higher-dimensional space.")
print("   It captures non-linear relationships better than linear models.")
print("   However, it is sensitive to scaling and hyperparameters.\n")

print("7. WHICH MODEL IS BEST (INTERPRETATION):")
print("   If Linear Regression R2 is low → data is non-linear.")
print("   If Decision Tree R2 is high but unstable → overfitting.")
print("   If SVM performs well → non-linear structure exists in data.\n")

print("8. FINAL INSIGHT:")
print("   Concrete strength is influenced by material chemistry + curing time.")
print("   The relationship is non-linear, so advanced models outperform simple linear models.")



# ==========================================
# CONCRETE CLASSIFICATION (FULL PIPELINE)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_excel("Concrete_Data.xls")

print("===== DATA =====")
print(df.head())

# ==========================================
# 2. CONVERT TO CLASSIFICATION
# ==========================================

# Target column
target = df.columns[-1]

# Create 3 classes based on strength
df['Strength_Class'] = pd.qcut(df[target], q=3, labels=[0,1,2])

print("\n===== CLASS DISTRIBUTION =====")
print(df['Strength_Class'].value_counts())

# ==========================================
# 3. FEATURES & TARGET
# ==========================================
X = df.iloc[:, :-2]   # exclude original target + new class
y = df['Strength_Class']

# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==========================================
# 5. SCALING (for SVM & Logistic)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. MODELS
# ==========================================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================
def evaluate_model(name, y_test, y_pred):
    print(f"\n===== {name} =====")
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ==========================================
# 8. RESULTS
# ==========================================
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("SVM", y_test, svm_pred)

# ==========================================
# 9. CONFUSION MATRIX VISUALIZATION
# ==========================================
def plot_cm(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(confusion_matrix(y_test, lr_pred), "LR Confusion Matrix")
plot_cm(confusion_matrix(y_test, dt_pred), "DT Confusion Matrix")
plot_cm(confusion_matrix(y_test, svm_pred), "SVM Confusion Matrix")

# ==========================================
# 10. DEEP INFERENCE (WHY-BASED)
# ==========================================

print("\n===== DETAILED INFERENCE =====\n")

print("1. WHY convert regression to classification?")
print("   Real-world decisions often need categories (low/medium/high strength).")
print("   It simplifies interpretation and decision-making in construction.\n")

print("2. WHY Logistic Regression works:")
print("   It assumes linear boundaries between classes.")
print("   Works well when class separation is simple.")
print("   But concrete strength relationships are complex → limits performance.\n")

print("3. WHY Decision Tree performs differently:")
print("   Splits data into rule-based regions.")
print("   Captures non-linear relationships easily.")
print("   But may overfit → performs very well on training, less on unseen data.\n")

print("4. WHY SVM performs strong:")
print("   Uses kernel trick (RBF) to separate complex patterns.")
print("   Finds optimal boundary with maximum margin.")
print("   Works well when classes are not linearly separable.\n")

print("5. WHY Accuracy alone is not enough:")
print("   Accuracy can be misleading if classes are imbalanced.")
print("   Example: Predicting only one class can still give high accuracy.\n")

print("6. WHY F1-score is important:")
print("   Combines Precision and Recall.")
print("   Useful when false positives and false negatives matter.\n")

print("7. WHY Confusion Matrix matters:")
print("   Shows exact classification errors.")
print("   Helps understand which class is being misclassified.\n")

print("8. FINAL INSIGHT:")
print("   Concrete strength classification is non-linear.")
print("   SVM and Decision Tree usually outperform Logistic Regression.")
print("   Model choice depends on accuracy vs interpretability trade-off.")