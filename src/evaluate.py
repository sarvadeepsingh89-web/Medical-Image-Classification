import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from config import TEST_DIR, IMG_SIZE, BATCH_SIZE


# 🔹 Load Test Dataset
def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='binary'
    )
    AUTOTUNE = tf.data.AUTOTUNE
    return test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# 🔹 Evaluation Function
def evaluate_model(model_path, model_name="Fine-tuned Model"):
    print(f"\n🚀 Starting evaluation for: {model_name}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

    # Load dataset & model
    test_ds = load_test_dataset()
    model = load_model(model_path, custom_objects={'preprocess_input': preprocess_input})
    print("✅ Model loaded successfully!")

    # Predictions
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds).ravel()
    y_pred_classes = (y_pred > 0.55).astype("int32")

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_classes, digits=2))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")

    # ✅ Save inside Kaggle's working directory
    output_dir = "/kaggle/working/assets"
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()  
    print(f"✅ Confusion matrix saved successfully at: {cm_path}")

    # ROC Curve (only for Fine-tuned model)
    if "feature" not in model_name.lower():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)

        roc_path = os.path.join(output_dir, f"roc_curve_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(roc_path, dpi=300)
        plt.show() 
        print(f"✅ ROC curve saved successfully at: {roc_path}")
    else:
        print("ℹ️ Skipping ROC curve for Feature Extraction model (only for fine-tuned model).")


# 🔹 Run Evaluations
if __name__ == "__main__":
    base_dir = "/kaggle/working/model" if os.path.exists("/kaggle/working") else "../models"
    evaluate_model(f"{base_dir}/best_resnet_head.keras", "Feature Extraction")
    evaluate_model(f"{base_dir}/final_resnet_pneumonia_model.keras", "Fine-tuned")
