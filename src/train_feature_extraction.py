import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import LEARNING_RATE, EPOCHS, BEST_HEAD_MODEL, CSV_LOG_PATH


def train_model(model, train_ds, val_ds, class_weights):
   
    # 🔹 Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 🔹 Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
        ModelCheckpoint(BEST_HEAD_MODEL, monitor="val_loss", save_best_only=True, mode="min"),
        tf.keras.callbacks.CSVLogger(CSV_LOG_PATH, append=True)
    ]

    # 🔹 Train Model
    print("\n🚀 Starting Feature Extraction Training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 🔹 Save Best Model
    model.save(BEST_HEAD_MODEL)
    print(f"\n💾 Model saved at: {BEST_HEAD_MODEL}")

    return history
