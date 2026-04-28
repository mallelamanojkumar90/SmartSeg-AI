import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import pickle
import os

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('data/Mall_Customers.csv')

    # 1. Data Preprocessing
    print("Preprocessing data...")
    # Drop CustomerID
    df_processed = df.drop('CustomerID', axis=1)
    
    # Encode Gender
    le = LabelEncoder()
    df_processed['Gender'] = le.fit_transform(df_processed['Gender'])
    
    # Scale features
    scaler = StandardScaler()
    features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    df_scaled = scaler.fit_transform(df_processed[features])
    
    # 2. Elbow Method
    print("Implementing Elbow Method...")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('outputs/elbow_plot.png')
    print("Elbow plot saved to outputs/elbow_plot.png")

    # 3. Train with optimal k (default 5 as requested)
    k = 5
    print(f"Training K-Means with k={k}...")
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # 4. Compute Silhouette Score
    score = silhouette_score(df_scaled, clusters)
    print(f"Silhouette Score for k={k}: {score:.4f}")

    # 5. Save model, scaler and encoder
    print("Saving model and tools...")
    with open('model/kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save the scaled data for visualization in app
    df_result = df_processed.copy()
    df_result['Cluster'] = clusters
    df_result.to_csv('data/clustered_customers.csv', index=False)
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()
