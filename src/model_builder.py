import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50
from config import L2_REG, _preprocess_input

# 🧱 Build Feature Extraction Model
def build_feature_extraction_model(data_augmentation):

    # Keep spatial feature maps (pooling=None)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling=None      # 🔥 Important for Grad-CAM
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = data_augmentation(inputs)
    x = layers.Lambda(_preprocess_input, name="preprocess")(x)
    x = base_model(x, training=False)                          
    x = layers.GlobalAveragePooling2D(name="gap")(x)           
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="resnet50_feature_extraction")
    return model, base_model

# 🧩 Build Fine-Tuning Model
def build_fine_tuning_model(model_path, data_augmentation):
    model = tf.keras.models.load_model(model_path)

    base_model = model.get_layer('resnet50')
    base_model.trainable = True

    fine_tune_at = len(base_model.layers) - 30
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= fine_tune_at)

    print(f"✅ Fine-tuning from layer {fine_tune_at} onward "
          f"({len(base_model.layers) - fine_tune_at} layers unfrozen)")

    return model, base_model

if __name__ == "__main__":
    print("Model Builder Module Loaded ✅")
