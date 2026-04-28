---
title: SmartSeg AI
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
---

# 🤖 SmartSeg AI – Advanced Customer Segmentation Dashboard

SmartSeg AI is a professional machine learning application that uses K-Means clustering to segment customers based on their demographic and spending behavior. It provides real-time predictions and interactive visualizations to help businesses understand their customer base better.

## 🚀 Features

- **K-Means Clustering:** Robust segmentation using industry-standard algorithms.
- **Real-time Prediction:** Instant classification of new customer data.
- **Interactive Dashboards:** Visualized using Plotly and Gradio.
- **PCA Analysis:** 2D projection of high-dimensional data for better cluster visibility.
- **Business Insights:** Actionable advice based on customer segments.

## 📊 Dataset

The project uses the **Mall Customer Segmentation** dataset, which includes:
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://huggingface.co/spaces/your-username/smartseg-ai
   cd smartseg-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional):**
   ```bash
   python train.py
   ```

4. **Run the App:**
   ```bash
   python app.py
   ```

## 📂 Project Structure

```text
smartseg-ai/
│
├── data/
│   ├── Mall_Customers.csv        # Raw dataset
│   └── clustered_customers.csv  # Data with cluster labels
├── model/
│   ├── kmeans.pkl               # Trained K-Means model
│   ├── scaler.pkl               # StandardScaler object
│   └── label_encoder.pkl        # LabelEncoder for Gender
├── outputs/
│   └── elbow_plot.png           # Elbow method visualization
├── train.py                     # Model training script
├── app.py                       # Gradio dashboard application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🤖 Hugging Face Deployment

This app is ready to be deployed to [Hugging Face Spaces](https://huggingface.co/spaces).
1. Create a new Space with the **Gradio** SDK.
2. Upload all files (including `model/` and `data/` directories).
3. The app will automatically build and run.

## 📸 Screenshots

*(Add screenshots here after running the app)*

---
Built with ❤️ by Manojkumar