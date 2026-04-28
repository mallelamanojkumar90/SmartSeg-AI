import gradio as gr
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os

# Load model, scaler, and label encoder
with open('model/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load clustered data for visualization
df_clustered = pd.read_csv('data/clustered_customers.csv')

# Cluster labels and descriptions
CLUSTER_MAP = {
    0: {"label": "Sensible Customers", "desc": "Medium income, medium spending. Stable group.", "insight": "Maintain relationship with loyalty programs."},
    1: {"label": "High Value Targets", "desc": "High income, high spending. Most profitable.", "insight": "Offer premium deals and exclusive previews."},
    2: {"label": "Budget Conscious", "desc": "Low income, low spending. Price sensitive.", "insight": "Focus on value-for-money offers and discounts."},
    3: {"label": "Impulsive Buyers", "desc": "Low income, high spending. High engagement.", "insight": "Target with flash sales and trendy items."},
    4: {"label": "Cautious High-Earners", "desc": "High income, low spending. Conservative.", "insight": "Focus on quality and long-term value marketing."}
}

def predict_and_visualize(gender, age, income, spending_score):
    # Prepare input for prediction
    gender_encoded = le.transform([gender])[0]
    input_data = np.array([[gender_encoded, age, income, spending_score]])
    input_scaled = scaler.transform(input_data)
    
    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]
    info = CLUSTER_MAP[cluster]
    
    # Update data with user input for visualization
    user_point = pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending_score],
        'Cluster': [cluster],
        'Type': ['User Input']
    })
    
    temp_df = df_clustered.copy()
    temp_df['Type'] = 'Existing Customer'
    
    # (A) Cluster Scatter Plot (Income vs Spending)
    fig_scatter = px.scatter(
        temp_df, x='Annual Income (k$)', y='Spending Score (1-100)',
        color='Cluster', symbol='Type',
        title='Customer Segments: Income vs Spending',
        color_continuous_scale='Viridis',
        labels={'Cluster': 'Segment ID'}
    )
    
    # Add user point
    fig_scatter.add_trace(
        go.Scatter(
            x=[income], y=[spending_score],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='white')),
            name='YOU'
        )
    )
    
    # (B) PCA Visualization
    pca = PCA(n_components=2)
    features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    all_features_scaled = scaler.transform(df_clustered[features])
    pca_results = pca.fit_transform(all_features_scaled)
    
    df_pca = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = df_clustered['Cluster']
    
    user_pca = pca.transform(input_scaled)
    
    fig_pca = px.scatter(
        df_pca, x='PC1', y='PC2',
        color='Cluster',
        title='PCA: 2D Cluster Separation',
        color_continuous_scale='Plasma'
    )
    
    fig_pca.add_trace(
        go.Scatter(
            x=[user_pca[0][0]], y=[user_pca[0][1]],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='white')),
            name='YOU'
        )
    )

    result_md = f"""
    ### 🎯 Prediction: {info['label']}
    **Description:** {info['desc']}
    
    **💡 Business Insight:** 
    *{info['insight']}*
    """
    
    return result_md, fig_scatter, fig_pca

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="SmartSeg AI") as demo:
    gr.Markdown("""
    # 🤖 SmartSeg AI – Advanced Customer Segmentation
    ### Analyze and categorize customers in real-time using Machine Learning.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Customer Profile")
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Female")
            age = gr.Slider(18, 100, step=1, label="Age", value=30)
            income = gr.Slider(10, 150, step=1, label="Annual Income (k$)", value=50)
            spending = gr.Slider(1, 100, step=1, label="Spending Score (1-100)", value=50)
            
            predict_btn = gr.Button("Predict Segment", variant="primary")
            
            output_text = gr.Markdown("### Results will appear here after prediction.")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Income vs Spending"):
                    plot_scatter = gr.Plot()
                with gr.TabItem("PCA 2D Projection"):
                    plot_pca = gr.Plot()

    # Define interaction
    predict_btn.click(
        fn=predict_and_visualize,
        inputs=[gender, age, income, spending],
        outputs=[output_text, plot_scatter, plot_pca]
    )
    
    # Initialize with a default prediction
    demo.load(predict_and_visualize, [gender, age, income, spending], [output_text, plot_scatter, plot_pca])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
