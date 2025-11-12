import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED

# 📦 Load Datasets
def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.15,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.15,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='binary'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='binary'
    )

    # ✅ Optimize pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print(f"✅ Datasets Loaded: {len(train_ds)} train batches, {len(val_ds)} val batches, {len(test_ds)} test batches.")
    return train_ds, val_ds, test_ds


# ⚖️ Compute Class Weights
def get_class_weights(train_ds):

    labels = np.concatenate([y.numpy().astype(int).reshape(-1,) for _, y in train_ds], axis=0)
    classes = np.unique(labels)
    class_weights_raw = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_raw)}

    print(f"✅ Computed class weights: {class_weights}")
    return class_weights

# 🧩 Data Augmentation Layer
def get_data_augmentation():
    
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.15),
    ], name="data_augmentation")

# 🧩 Load Only Test Dataset (for Grad-CAM and Evaluation)
def load_test_dataset():
 
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='binary'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    print(f"✅ Test dataset loaded successfully: {len(test_ds)} batches.")
    return test_ds


# 🧪 Test Run 
if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()
    class_weights = get_class_weights(train_ds)
    print("Data loader test completed ✅")
