import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. LOAD THE DATA
# Define columns based on the dataset documentation [cite: 14, 21]
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

try:
    # Loading 'iris.data' as is [cite: 22]
    df = pd.read_csv('iris.data', names=columns)
    print("Dataset loaded successfully!")

    # 2. MACHINE LEARNING SETUP
    X = df.drop('species', axis=1)
    y = df['species']

    # The dataset contains 150 instances (50 per class) [cite: 20]
    # Splitting into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TRAIN MODEL
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. PREDICTIONS
    y_pred = model.predict(X_test)

    # 5. ALL METRICS
    print("\n--- CLASSIFICATION REPORT ---")
    # Shows Precision, Recall, and F1-Score for Setosa, Versicolour, and Virginica [cite: 21]
    print(classification_report(y_test, y_pred))

    print("--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 6. VISUALIZATION
    # Confusion Matrix visualizes misclassification rates [cite: 8, 10]
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix: Iris Species Classification")
    plt.show()

except FileNotFoundError:
    print("Error: 'iris.data' not found. Please ensure the file is in your VS Code folder.")
except Exception as e:
    print(f"An error occurred: {e}")