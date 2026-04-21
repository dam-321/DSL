# ===============================================================
# COMPLETE CLUSTERING MODEL LAB CODE
# Walmart Sales Dataset
# Includes:
# 1. Data Loading
# 2. Preprocessing
# 3. Descriptive Analysis
# 4. Elbow Method
# 5. KMeans Clustering
# 6. Hierarchical Clustering
# 7. DBSCAN
# 8. Gaussian Mixture Model
# 9. Evaluation Metrics
# 10. Cluster Visualization
# ===============================================================


# ===============================================================
# STEP 1 : IMPORT LIBRARIES
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings("ignore")


# ===============================================================
# STEP 2 : LOAD DATASET
# ===============================================================

df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")

print("First 5 Rows")
print(df.head())

print("\nShape :", df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())


# ===============================================================
# STEP 3 : DATA PREPROCESSING
# ===============================================================

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Extract Date Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Fill Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)


# ===============================================================
# STEP 4 : SELECT FEATURES FOR CLUSTERING
# ===============================================================

X = df[['Store',
        'Weekly_Sales',
        'Holiday_Flag',
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment',
        'Month']]

print("\nSelected Features")
print(X.head())


# ===============================================================
# STEP 5 : FEATURE SCALING
# ===============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ===============================================================
# STEP 6 : DESCRIPTIVE VISUALIZATION
# ===============================================================

plt.figure(figsize=(8,5))
sns.histplot(df['Weekly_Sales'], kde=True)
plt.title("Weekly Sales Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ===============================================================
# STEP 7 : ELBOW METHOD (Find Best K)
# ===============================================================

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()

# Assume K = 3


# ===============================================================
# STEP 8 : KMEANS CLUSTERING
# ===============================================================

kmeans = KMeans(n_clusters=3, random_state=42)

df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print("\nKMeans Cluster Counts")
print(df['KMeans_Cluster'].value_counts())

print("Silhouette Score :",
      silhouette_score(X_scaled, df['KMeans_Cluster']))

print("Davies Bouldin Score :",
      davies_bouldin_score(X_scaled, df['KMeans_Cluster']))

print("Calinski Harabasz Score :",
      calinski_harabasz_score(X_scaled, df['KMeans_Cluster']))


# Visualization
plt.figure(figsize=(10,6))
plt.scatter(df['Store'],
            df['Weekly_Sales'],
            c=df['KMeans_Cluster'],
            cmap='viridis')

plt.title("KMeans Clustering")
plt.xlabel("Store")
plt.ylabel("Weekly Sales")
plt.show()


# ===============================================================
# STEP 9 : HIERARCHICAL CLUSTERING
# ===============================================================

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12,6))
dendrogram(linked)
plt.title("Hierarchical Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

hc = AgglomerativeClustering(n_clusters=3)

df['Hierarchical_Cluster'] = hc.fit_predict(X_scaled)

print("\nHierarchical Cluster Counts")
print(df['Hierarchical_Cluster'].value_counts())

print("Silhouette Score :",
      silhouette_score(X_scaled, df['Hierarchical_Cluster']))

plt.figure(figsize=(10,6))
plt.scatter(df['Store'],
            df['Weekly_Sales'],
            c=df['Hierarchical_Cluster'],
            cmap='rainbow')

plt.title("Hierarchical Clustering")
plt.xlabel("Store")
plt.ylabel("Weekly Sales")
plt.show()


# ===============================================================
# STEP 10 : DBSCAN CLUSTERING
# ===============================================================

db = DBSCAN(eps=1.5, min_samples=5)

df['DBSCAN_Cluster'] = db.fit_predict(X_scaled)

print("\nDBSCAN Unique Clusters")
print(np.unique(df['DBSCAN_Cluster']))

plt.figure(figsize=(10,6))
plt.scatter(df['Store'],
            df['Weekly_Sales'],
            c=df['DBSCAN_Cluster'],
            cmap='plasma')

plt.title("DBSCAN Clustering")
plt.xlabel("Store")
plt.ylabel("Weekly Sales")
plt.show()


# ===============================================================
# STEP 11 : GAUSSIAN MIXTURE MODEL
# ===============================================================

gmm = GaussianMixture(n_components=3, random_state=42)

df['GMM_Cluster'] = gmm.fit_predict(X_scaled)

print("\nGMM Cluster Counts")
print(df['GMM_Cluster'].value_counts())

print("Silhouette Score :",
      silhouette_score(X_scaled, df['GMM_Cluster']))

plt.figure(figsize=(10,6))
plt.scatter(df['Store'],
            df['Weekly_Sales'],
            c=df['GMM_Cluster'],
            cmap='Set1')

plt.title("Gaussian Mixture Clustering")
plt.xlabel("Store")
plt.ylabel("Weekly Sales")
plt.show()


# ===============================================================
# STEP 12 : CLUSTER ANALYSIS
# ===============================================================

print("\nAverage Weekly Sales by KMeans Cluster")
print(df.groupby('KMeans_Cluster')['Weekly_Sales'].mean())

print("\nAverage Temperature by Cluster")
print(df.groupby('KMeans_Cluster')['Temperature'].mean())


# ===============================================================
# STEP 13 : FINAL COMPARISON
# ===============================================================

models = ['KMeans','Hierarchical','GMM']
scores = [
    silhouette_score(X_scaled, df['KMeans_Cluster']),
    silhouette_score(X_scaled, df['Hierarchical_Cluster']),
    silhouette_score(X_scaled, df['GMM_Cluster'])
]

plt.figure(figsize=(8,5))
plt.bar(models, scores)
plt.title("Silhouette Score Comparison")
plt.ylabel("Score")
plt.show()


# ===============================================================
# STEP 14 : SAVE OUTPUT FILE
# ===============================================================

df.to_csv("clustered_walmart_output.csv", index=False)

print("\nClustered dataset saved successfully.")


# ===============================================================
# END OF COMPLETE CLUSTERING CODE
# ===============================================================