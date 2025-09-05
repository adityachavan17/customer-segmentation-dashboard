# 🛍️ Customer Segmentation Dashboard

An end-to-end data science application that segments e-commerce customers using **RFM analysis** and **K-Means clustering**, with results visualized in an **interactive Streamlit dashboard** and stored in **MongoDB Atlas** for persistence.

🎯 **Goal**: Help businesses identify high-value, at-risk, and loyal customers for targeted marketing and retention strategies.

---

## 🔧 Features

- 📤 Upload a CSV file of transaction data  
- 🧹 Automatic data cleaning (handles canceled orders, missing IDs, invalid dates)  
- 📊 RFM Analysis (Recency, Frequency, Monetary)  
- 🧠 K-Means Clustering with Elbow Method to find optimal segments  
- 📈 PCA Visualization of clusters in 2D  
- 🧑‍💼 Segment Profiling with meaningful names:
  - `High Value, Active`
  - `Frequent Buyers`
  - `New Customers`
  - `Potential Loyalists`
  - `High Value, Inactive`
  - `Low Engagement`
  - `Lost Customers`
  - `High Spending`
  - `Average Customers`
- 📋 View and download customer lists by segment  
- ☁️ Save & Load segmented data from **MongoDB Atlas (cloud database)**  
- 🖼️ Interactive Streamlit dashboard with clean, modern UI  

---

## 📸 Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Cluster Visualization
![Clusters](screenshots/clustering.png)

### Customer Segments
![Segments](screenshots/segmentation.png)



---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

Activate it:  
**Windows**:
```bash
venv\Scripts\activate
```
**macOS/Linux**:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your MongoDB Credentials
Create a `.env` file in the root folder:

```env
MONGODB_URI=mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
DB_NAME=customer_segments_db
COLLECTION_NAME=customers_rfm_segments
```

🔐 Replace with your actual MongoDB Atlas credentials.  
⚠️ **Never commit or share this file publicly.**

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deployed Version (Optional)

🌐 Live Demo: https://yourname-customer-segmentation.streamlit.app  

🎯 You can deploy your own version for free on **Streamlit Community Cloud**.

---

## 🛠️ Tech Stack

- **Python** – Core logic  
- **Pandas** – Data loading and transformation  
- **Scikit-learn** – K-Means clustering, PCA, StandardScaler  
- **Streamlit** – Interactive dashboard  
- **Matplotlib & Seaborn** – Data visualization  
- **MongoDB Atlas** – Cloud database for persistent storage  
- **Python-dotenv** – Secure handling of environment variables  

---

## 🎯 Why This Project Stands Out

This project demonstrates a full data science pipeline from raw data to actionable insights:

- 🧹 Data Cleaning – Handle real-world issues like canceled orders and missing values  
- 🔑 Feature Engineering – Compute RFM scores from transaction history  
- 🤖 Machine Learning – Apply K-Means clustering (unsupervised learning)  
- 🔍 Dimensionality Reduction – Use PCA to visualize clusters  
- ☁️ Database Skills – Store results in MongoDB Atlas (cloud)  
- 📊 Dashboarding – Build a user-friendly interface with Streamlit  
- 🚀 Deployment Ready – Can be hosted online for anyone to use  

Perfect for **data analyst, data scientist, or business analyst** roles.

---

## 📂 Project Structure

```
customer-segmentation/
├── app.py                  # Main Streamlit dashboard
├── utils.py                # Data cleaning, RFM, clustering logic
├── database.py             # MongoDB save/load functions
├── .env                    # Environment variables (not committed)
├── requirements.txt        # Python dependencies
├── data/                   # Folder for CSV file
│   └── online_retail.csv
├── screenshots/            # Screenshots for README
│   ├── dashboard.png
│   ├── clustering.png
│   └── segments.png
├── .gitignore              # Ignored files (venv, .env, cache)
└── README.md               # This file
```

---

## 🙌 Acknowledgements

This project was built as a **learning exercise** to demonstrate end-to-end customer segmentation using industry-standard techniques.

The methodology follows best practices in:

- Customer analytics (RFM)  
- Unsupervised learning (K-Means)  
- Data visualization (PCA)  
- Cloud storage (MongoDB Atlas)  
- Interactive dashboards (Streamlit)  

All code was written from scratch for educational purposes.

---
