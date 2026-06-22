# 🛒 ShopperSpectrum: Customer Segmentation and Product Recommendation System

An end-to-end Machine Learning project that combines **Customer Segmentation using RFM Analysis and K-Means Clustering** with a **Product Recommendation System based on Collaborative Filtering**.

## 🚀 Live Demo

🔗 Streamlit App: YOUR_STREAMLIT_LINK_HERE

---

## 📌 Project Overview

ShopperSpectrum helps businesses:

- Identify different customer groups based on purchasing behavior.
- Recommend similar products to customers.
- Improve customer retention.
- Enable personalized marketing strategies.

---

## 🎯 Features

### 📊 Customer Segmentation
- RFM Analysis
- K-Means Clustering
- Customer classification into:
  - 🌟 High Value
  - 🙂 Regular
  - 🛍 Occasional
  - ⚠️ At Risk

### 🎁 Product Recommendation System
- Item-based Collaborative Filtering
- Top 5 similar product recommendations
- Similarity scores displayed

### 🌐 Interactive Streamlit Web App
- Clean UI
- Real-time predictions
- Product recommendation module
- Customer segmentation module

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

## 📂 Project Structure

```text
ShopperSpectrum
│
├── data
│   ├── cleaned_online_retail.csv
│   └── customer_segments.csv
│
├── models
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   └── recommendations.pkl
│
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_customer_segmentation.ipynb
│   ├── 04_product_recommendation.ipynb
│   └── 05_model_evaluation.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📈 Machine Learning Workflow

### 1️⃣ Data Preprocessing
- Handling missing values
- Removing cancelled transactions
- Feature creation

### 2️⃣ Exploratory Data Analysis
- Top products
- Top countries
- Revenue analysis
- Monthly sales trends

### 3️⃣ Customer Segmentation
- RFM Feature Engineering
- Log Transformation
- Standardization
- Elbow Method
- Silhouette Score
- K-Means Clustering

### 4️⃣ Product Recommendation
- Customer-product matrix
- Cosine similarity
- Top-N recommendations

### 5️⃣ Streamlit Deployment
- Interactive web application
- Real-time predictions

---

## 📊 Customer Segments

| Segment | Description |
|-----------|------------|
| 🌟 High Value | Frequent and high-spending customers |
| 🙂 Regular | Steady customers with moderate spending |
| 🛍 Occasional | Infrequent buyers |
| ⚠️ At Risk | Customers inactive for a long time |

---

## 💡 Business Applications

- Personalized marketing
- Cross-selling opportunities
- Customer retention
- Product recommendation
- Customer behavior analysis

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/akshvjk13/ShopperSpectrum.git
```

Move into the project folder:

```bash
cd ShopperSpectrum
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## 📷 Screenshots

(Add screenshots of your Streamlit app here)

---

## 👩‍💻 Developed By

**Akshitha Chunchu**

Third Year Computer Engineering Student  
Thakur College of Engineering and Technology (TCET)

---

## ⭐ If you found this project useful, consider giving it a star!