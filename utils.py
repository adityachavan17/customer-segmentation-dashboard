# utils.py
# Core logic: Load, clean, RFM, scale, cluster, PCA, segment naming

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def log(msg):
    """Simple print function for logging"""
    print(msg)


# --- 1. Load and clean data ---
def load_and_clean_data(df):
    """
    Clean transaction data: remove canceled orders, missing IDs, invalid dates
    Input: DataFrame (from uploaded CSV)
    Output: Cleaned DataFrame
    """
    log("🧹 Starting data cleaning...")
    data = df.copy()

    # Remove canceled orders (InvoiceNo starts with 'C')
    data['InvoiceNo_str'] = data['InvoiceNo'].astype(str)
    canceled_mask = data['InvoiceNo_str'].str.startswith('C')
    num_canceled = canceled_mask.sum()
    log(f"🗑️ Removed {num_canceled} canceled orders")
    data = data[~canceled_mask]
    data.drop(columns=['InvoiceNo_str'], inplace=True)

    # Drop missing CustomerID
    log(f"⚠️ Missing CustomerID: {data['CustomerID'].isnull().sum()} rows")
    data = data.dropna(subset=['CustomerID'])
    data['CustomerID'] = data['CustomerID'].astype(int)

    # Parse InvoiceDate (handles DD-MM-YYYY)
    log("📅 Converting InvoiceDate to datetime...")
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=True, errors='coerce')
    data = data.dropna(subset=['InvoiceDate'])

    # Create TotalAmount
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']

    # Keep only valid transactions
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0) & (data['TotalAmount'] > 0)]

    data = data.reset_index(drop=True)
    log(f"✅ Data cleaning done! {len(data)} transactions remain.")
    return data


# --- 2. Compute RFM ---
def compute_rfm(data):
    """
    Compute Recency, Frequency, Monetary
    """
    log("🧮 Computing RFM scores...")
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    latest_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    log(f"🎯 Reference date: {latest_date.date()}")

    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).round(2)

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    log("✅ RFM table created!")
    return rfm


# --- 3. Scale RFM ---
def scale_rfm(rfm_df):
    """
    Scale RFM data for clustering
    """
    log("⚖️ Scaling RFM data...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    log("✅ RFM scaled!")
    return rfm_scaled, scaler


# --- 4. K-Means Clustering ---
def apply_kmeans_clustering(rfm_scaled):
    """
    Apply K-Means and show Elbow plot
    """
    log("🎯 Applying K-Means clustering...")

    k_range = range(1, 11)
    inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)

    # Save Elbow Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.savefig("elbow_plot.png")
    plt.close()
    log("📉 Elbow plot saved as 'elbow_plot.png'")

    optimal_k = 4
    log(f"Using K = {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    log(f"✅ Clustering done! {optimal_k} clusters created.")
    return kmeans, labels


# --- 5. PCA for Visualization ---
def apply_pca(rfm_scaled):
    """
    Reduce to 2D for scatter plot
    """
    log("📉 Applying PCA...")
    pca = PCA(n_components=2, random_state=42)
    rfm_pca = pca.fit_transform(rfm_scaled)
    log(f"✅ PCA done! Explained variance: {pca.explained_variance_ratio_}")
    return rfm_pca


# --- 6. Assign Segment Names ---
def assign_cluster_names(rfm_df_with_cluster):
    """
    Assign simple, neutral, and professional names to clusters.
    Input: RFM DataFrame with 'Cluster' column
    Output: DataFrame with 'Segment' column and mapping
    """
    print("🏷️ Assigning simple and neutral segment names...")

    # Make a copy to avoid modifying original
    data = rfm_df_with_cluster.copy()
    
    # Ensure CustomerID is a column
    if 'CustomerID' not in data.columns:
        data = data.reset_index()

    # Compute average R, F, M per cluster
    cluster_summary = data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Mapping from cluster number to simple name
    cluster_names = {}

    for cluster in sorted(cluster_summary.index):
        r = cluster_summary.loc[cluster, 'Recency']
        f = cluster_summary.loc[cluster, 'Frequency']
        m = cluster_summary.loc[cluster, 'Monetary']

        # Define rules based on RFM averages
        if r < 30 and f > 10 and m > 5000:
            name = "High Value, Active"
        elif r < 30 and f > 5:
            name = "Frequent Buyers"
        elif r < 30 and f <= 5:
            name = "New Customers"
        elif r >= 30 and r < 90 and f > 5:
            name = "Potential Loyalists"
        elif r >= 30 and f > 10 and m > 10000:
            name = "High Value, Inactive"
        elif r >= 90 and f > 2:
            name = "Low Engagement"
        elif r >= 90 and f == 1:
            name = "Lost Customers"
        elif f > 10 and m > 10000:
            name = "High Spending"
        else:
            name = "Average Customers"

        cluster_names[cluster] = name

    # Add the formal segment name to each customer
    data['Segment'] = data['Cluster'].map(cluster_names)

    return data, cluster_names