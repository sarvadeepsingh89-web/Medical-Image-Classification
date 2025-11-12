import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from data_loader import load_datasets, get_class_weights, get_data_augmentation
from model_builder import build_fine_tuning_model
from config import BEST_HEAD_MODEL, LEARNING_RATE, EPOCHS, CSV_LOG_PATH


def finetune_model(model, base_model, train_ds, val_ds, class_weights):

    # 🔹 Compile with smaller LR for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 🔹 Prepare directories
    MODEL_DIR = os.path.join("/kaggle/working/model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_finetuned_model = os.path.join(MODEL_DIR, "best_model_finetuned.keras")
    final_model_path = os.path.join(MODEL_DIR, "final_resnet_pneumonia_model.keras")

    # 🔹 Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
        ModelCheckpoint(best_finetuned_model, monitor="val_loss", save_best_only=True, mode="min"),
        tf.keras.callbacks.CSVLogger(CSV_LOG_PATH, append=True)
    ]

    # 🔹 Start Training
    print("\n🚀 Starting Fine-Tuning Training...\n")
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 🔹 Save final fine-tuned model
    model.save(final_model_path)
    print(f"💾 Final fine-tuned model saved at: {final_model_path}")

    return history_finetune


# 🧩 Safe Entry Point 
if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()
    class_weights = get_class_weights(train_ds)
    data_augmentation = get_data_augmentation()
    model, base_model = build_fine_tuning_model(BEST_HEAD_MODEL, data_augmentation)
    finetune_model(model, base_model, train_ds, val_ds, class_weights)
