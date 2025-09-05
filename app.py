# app.py
# Customer Segmentation Dashboard

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_and_clean_data,
    compute_rfm,
    scale_rfm,
    apply_kmeans_clustering,
    apply_pca,
    assign_cluster_names
)
from database import save_rfm_to_mongodb, load_rfm_from_mongodb


# --- Page Setup ---
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation Dashboard")
st.markdown("Upload data → Segment customers → Save to cloud")


# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Choose Action")
    option = st.radio("Go to", [
        "1. Upload & Clean",
        "2. Compute RFM",
        "3. Cluster Customers",
        "4. View Clusters",
        "5. Segment Profiling",
        "6. Save/Load"
    ])

# --- Session State ---
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'rfm_df' not in st.session_state:
    st.session_state.rfm_df = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'rfm_scaled' not in st.session_state:
    st.session_state.rfm_scaled = None
if 'rfm_pca' not in st.session_state:
    st.session_state.rfm_pca = None


# --- 1. Upload & Clean ---
if option == "1. Upload & Clean":
    st.header("📤 Upload Data")
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        try:
            df = pd.read_csv(uploaded, encoding='latin1')
            st.write("📋 First 5 rows:")
            st.write(df.head())

            with st.spinner("Cleaning..."):
                cleaned = load_and_clean_data(df)
            st.session_state.clean_data = cleaned
            st.success(f"✅ Cleaned data: {len(cleaned)} transactions")
        except Exception as e:
            st.error(f"Error: {e}")


# --- 2. Compute RFM ---
elif option == "2. Compute RFM":
    st.header("🧮 Compute RFM")

    if st.session_state.clean_data is not None:
        if st.button("Calculate RFM"):
            with st.spinner("Computing..."):
                rfm = compute_rfm(st.session_state.clean_data)
            st.session_state.rfm_df = rfm
            st.write("✅ RFM Table:")
            st.write(rfm.head())
    else:
        st.warning("Upload data first.")


# --- 3. Cluster ---
elif option == "3. Cluster Customers":
    st.header("🎯 Cluster Customers")

    if st.session_state.rfm_df is not None:
        if st.button("Run Clustering"):
            rfm_df = st.session_state.rfm_df

            with st.spinner("Scaling..."):
                rfm_scaled, _ = scale_rfm(rfm_df)
            st.session_state.rfm_scaled = rfm_scaled

            with st.spinner("Clustering..."):
                model, labels = apply_kmeans_clustering(rfm_scaled)
            st.session_state.cluster_labels = labels
            rfm_df['Cluster'] = labels
            st.session_state.rfm_df = rfm_df

            with st.spinner("PCA..."):
                rfm_pca = apply_pca(rfm_scaled)
            st.session_state.rfm_pca = rfm_pca

            st.success("✅ Clustering complete!")
            st.image("elbow_plot.png", caption="Elbow Method (K=4)")

    else:
        st.warning("Compute RFM first.")


# --- 4. View Clusters ---
elif option == "4. View Clusters":
    st.header("📈 Cluster Visualization")

    if st.session_state.rfm_pca is not None and st.session_state.rfm_df is not None:
        pca_df = pd.DataFrame(st.session_state.rfm_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = st.session_state.rfm_df['Cluster'].values

        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, ax=ax)
        ax.set_title("Customer Clusters (PCA)")
        st.pyplot(fig)

        st.write("📊 Cluster Distribution:")
        st.write(st.session_state.rfm_df['Cluster'].value_counts().sort_index())
    else:
        st.warning("Run clustering first.")


# --- 5. Segment Profiling ---
elif option == "5. Segment Profiling":
    st.header("🏷️ Segment Names")

    if st.session_state.rfm_df is not None and 'Cluster' in st.session_state.rfm_df.columns:
        with st.spinner("Naming segments..."):
            named_df, names = assign_cluster_names(st.session_state.rfm_df)
        st.session_state.named_df = named_df

        st.write("### 🧩 Segment Mapping")
        name_df = pd.DataFrame(list(names.items()), columns=['Cluster', 'Name'])
        st.write(name_df)

        profile = named_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
        profile['Count'] = named_df.groupby('Segment').size()
        st.write("### 📊 Segment Profile")
        st.write(profile)
    else:
        st.warning("Run clustering first.")


# --- 6. Save/Load ---
elif option == "6. Save/Load":
    st.header("💾 Save to MongoDB")

    if st.session_state.rfm_df is not None:
        if st.button("📤 Save to Cloud"):
            with st.spinner("Saving..."):
                success = save_rfm_to_mongodb(st.session_state.rfm_df)
            if success:
                st.success("✅ Saved to MongoDB Atlas!")
            else:
                st.error("❌ Failed. Check .env file.")

    if st.button("📥 Load from Cloud"):
        with st.spinner("Loading..."):
            loaded = load_rfm_from_mongodb()
        if loaded is not None:
            st.session_state.rfm_df = loaded
            st.success("✅ Loaded from MongoDB!")
            st.write(loaded.head())


# --- Footer ---
st.markdown("---")
st.caption("🎯 Customer Segmentation Dashboard | Built with Streamlit & MongoDB")