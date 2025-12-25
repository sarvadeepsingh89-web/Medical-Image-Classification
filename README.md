🩺 Pneumonia Detection using Deep Learning (CNN + Transfer Learning + Streamlit)
📘 Overview

This project focuses on detecting Pneumonia from chest X-ray images using Deep Learning techniques.
It combines a Baseline CNN model built from scratch and fine-tuned Transfer Learning models (e.g., ResNet50) to achieve high accuracy in identifying pneumonia cases.

A Streamlit web application is built to allow users to upload X-ray images and get real-time predictions.
This system aims to provide fast, automated, and reliable diagnostic support for healthcare professionals.

🧠 Project Workflow

1. Data Loading & Preprocessing:
   Loaded chest X-ray images and performed normalization, resizing, and augmentation using ImageDataGenerator.

2. Model Building:

   Baseline CNN: A simple CNN trained from scratch to establish baseline performance.

   ResNet50 (Transfer Learning): Fine-tuned on pneumonia dataset for better generalization and higher accuracy.

3. Model Evaluation:

   Visualized training and validation accuracy/loss curves.

   Generated Confusion Matrix, ROC Curve, and Grad-CAM heatmaps to interpret model decisions.

4. Model Demonstration:

   Built a Streamlit application for real-time medical image classification.   

   Users can upload a chest X-ray image and view the predicted class and confidence score.

🗂️ Folder Structure
Medical-Image-Classification/
│
├── assets/
│   ├── confusion_matrix.png
│   ├── gradcam_normal.png
│   ├── gradcam_pneumonia.png
│   ├── misclassified_images.png
│   └── roc_curve.png
│
├── model/
│   └── pneumonia_model.keras              # Saved trained model (Keras format)
│
├── notebooks/
│   ├── 01_baseline_cnn_from_scratch_colab.ipynb
│   ├── 02_resnet50_transfer_learning_kaggle.ipynb
│   └── 03_modular_pipeline_demo_kaggle.ipynb
│
├── src/
│   ├── config.py                          # Contains global configuration constants (paths, hyperparameters)
│   ├── data_loader.py                     # Loads and preprocesses training/validation/test datasets
│   ├── evaluate.py                        # Evaluates model performance and generates metrics
│   ├── gradcam_visualizer.py              # Generates Grad-CAM heatmaps for model interpretability
│   ├── model_builder.py                   # Builds CNN/ResNet model architectures
│   ├── train_feature_extraction.py        # Trains model with frozen base (feature extraction stage)
│   ├── train_fine_tune.py                 # Fine-tunes model layers for improved accuracy
│
├── app.py                                 # Streamlit app entry point
│
├── requirements.txt                       # Project dependencies
│
└── README.md                              # Project documentation (this file)


⚙️ Installation & Setup

Run the following commands to set up the project locally:

# 1️ Clone the repository
git clone https://github.com/sarvadeepsingh89-web/Medical-Image-Classification.git

# 2️ Navigate to the project directory
cd Medical-Image-Classification

# 3️ Install dependencies
pip install -r requirements.txt

# 4 Dockerization & Deployment  

This project has been containerized using Docker to ensure consistent execution across different environments.

The Streamlit application, along with all dependencies and the trained deep learning model, is packaged into a Docker image and published on Docker Hub.

Docker image:
https://hub.docker.com/r/sarvadeepsingh123/pneumonia-detection-cnn

🔹 Steps to Run Using Docker
# Pull the Docker image from Docker Hub
docker pull sarvadeepsingh123/pneumonia-detection-cnn:v1

# Run the container
docker run -p 8501:8501 sarvadeepsingh123/pneumonia-detection-cnn:v1

Once the container is running, open the browser and visit:
http://localhost:8501

# 4️ Run the Streamlit app
streamlit run app.py

🧩 Technologies Used

Python 3.x
TensorFlow / Keras
NumPy
Matplotlib & Seaborn
OpenCV
scikit-learn
Streamlit

#📊 Model Performance
Model	                        Accuracy	  Validation     Accuracy Loss	Remarks
Baseline CNN (Scratch)	        ~80.72%	   ~80.29%	   0.76	       Good starting point
ResNet50 (Transfer Learning)	  ~97%	      ~95%	      0.34	       Fine-tuned model gave best results

Highlights:

Transfer learning significantly improved accuracy and reduced overfitting.
ROC curve shows strong separation between Normal and Pneumonia classes.
Grad-CAM visualizations confirm model focuses on correct lung regions.

🧠 Key Insights

Pneumonia-infected lungs show red, yellow cloudy patches, while normal lungs appear clearer and darker.

The CNN model efficiently learned texture and density differences between both classes.
Data Augmentation and Batch Normalization helped reduce overfitting.
ResNet50 fine-tuning further improved classification reliability.

📸 Visualization Samples
Visualization	Description
🧩 confusion_matrix.png	Shows true vs predicted class distribution
🔥 gradcam_normal.png	Grad-CAM visualization for normal lungs
⚠️ gradcam_pneumonia.png	Grad-CAM visualization highlighting infected regions
❌ misclassified_images.png	Examples of incorrect predictions
📈 roc_curve.png	ROC curve showing AUC performance
🌐 Streamlit App

Features:

Upload a chest X-ray image (.jpg, .png, .jpeg)
Get prediction → “Normal” or “Pneumonia”
View confidence score
Optionally display Grad-CAM heatmap

🏗️ Future Enhancements

Integrate EfficientNet or DenseNet for higher accuracy.
Deploy using FastAPI or Flask REST API for broader access.
Add a dashboard to analyze model predictions over time.
Convert model to TensorFlow Lite for mobile deployment.

👨‍💻 Developer

Sarvadeep Singh
🔗 GitHub Profile

🏁 Conclusion

This project demonstrates how Deep Learning can support medical diagnostics by accurately identifying pneumonia from chest X-ray images.
With advanced CNN architectures, visual explainability tools (Grad-CAM), and a Streamlit interface, this project bridges the gap between AI and healthcare applications.

📎 Repository Link

👉 https://github.com/sarvadeepsingh89-web/Medical-Image-Classification
