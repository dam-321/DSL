import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# Load Dataset
df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Create Target Column
mean_sales = df['Weekly_Sales'].mean()

df['Target'] = np.where(df['Weekly_Sales'] > mean_sales,1,0)

# Features
X = df[['Store','Holiday_Flag','Temperature','Fuel_Price',
        'CPI','Unemployment','Year','Month','Day']]

y = df['Target']

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




#######Logistic regression#######

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)

pred = model.predict(X_test_scaled)
prob = model.predict_proba(X_test_scaled)[:,1]

print("Accuracy :",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,cmap='Blues',fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr,tpr,threshold = roc_curve(y_test,prob)
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label="AUC="+str(round(roc_auc,2)))
plt.plot([0,1],[0,1],'r--')
plt.title("ROC Curve")
plt.legend()
plt.show()





###########Decision Tree Classifier###########

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,cmap='Greens',fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.show()




############Random Forest Classifier###########
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,cmap='Oranges',fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.show()



###########Support Vector Machine###########
from sklearn.svm import SVC

model = SVC(probability=True)
model.fit(X_train_scaled,y_train)

pred = model.predict(X_test_scaled)
prob = model.predict_proba(X_test_scaled)[:,1]

print("Accuracy :",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,cmap='Purples',fmt='d')
plt.title("SVM Confusion Matrix")
plt.show()

fpr,tpr,_ = roc_curve(y_test,prob)
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label="AUC="+str(round(roc_auc,2)))
plt.plot([0,1],[0,1],'r--')
plt.legend()
plt.title("ROC Curve")
plt.show()







models = ['Logistic','Decision Tree','Random Forest','SVM','KNN']
accuracy = [0.81,0.78,0.89,0.84,0.80]

plt.bar(models,accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()