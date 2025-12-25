import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import tempfile
import os
import base64
import time

# ⚙️ App Config
st.set_page_config(page_title="Pneumonia Detection App", layout="wide")
st.title("🩺 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest x-ray image to predict if it shows **Pneumonia** or **Normal** lungs.")

# 📦 Load Model
@tf.keras.utils.register_keras_serializable(package="Custom")
def _preprocess_input(x):
    return preprocess_input(x)

@st.cache_resource
def load_pneumonia_model():
    model_path = "model/final_resnet_pneumonia_model.keras"
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"_preprocess_input": _preprocess_input},
        safe_mode=False
    )
    return model

model = load_pneumonia_model()

def generate_gradcam(model, img_array, save_path="GradCAM_Final.png"):
    import tensorflow as tf, numpy as np, cv2, os

    # --- Normalize input shape
    if isinstance(img_array, np.ndarray):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    else:
        img_tensor = tf.cast(img_array, tf.float32)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)

    # --- Find ResNet backbone
    try:
        resnet_layer = next(l for l in model.layers if "resnet" in l.name.lower())
    except StopIteration:
        raise RuntimeError("❌ Could not find ResNet backbone in model.layers.")

    x = img_tensor
    if "data_augmentation" in [l.name for l in model.layers]:
        x = model.get_layer("data_augmentation")(x, training=False)
    if "preprocess" in [l.name for l in model.layers]:
        x = model.get_layer("preprocess")(x)
    img_pre = x  

    print("img_tensor stats AFTER preprocess:", float(tf.reduce_min(img_pre).numpy()), float(tf.reduce_max(img_pre).numpy()))

    try:
        last_conv_name = resnet_layer.get_layer("conv5_block3_out").name
    except:
        for l in reversed(resnet_layer.layers):
            if isinstance(l, tf.keras.layers.Conv2D):
                last_conv_name = l.name
                break

    conv_tensor = resnet_layer.get_layer(last_conv_name).output
    conv_model = tf.keras.Model(inputs=resnet_layer.input, outputs=conv_tensor)

    # --- Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_pre, training=False)
        tape.watch(conv_outputs)

        x = tf.reduce_mean(conv_outputs, axis=(1, 2))

        after_backbone = False
        for layer in model.layers:
            if not after_backbone:
                if layer is resnet_layer:
                    after_backbone = True
                continue
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
         
            if layer.name in ["dropout", "dense", "dropout_1", "output"]:
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)

        preds = x
        loss = preds[:, 0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("❌ Gradients are None — check connectivity of conv maps and head layers.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    orig = img_tensor[0].numpy()
    if orig.max() <= 1.0:
        img_uint8 = (orig * 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(orig, 0, 255).astype(np.uint8)

    
    heatmap = np.power(heatmap, 2.2)

    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return save_path

# 🧾 Prediction Function 
def predict_image(model, image):
    """Return (pred_class_str, prob, preprocessed_img, display_img_uint8)."""
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_pre = img_array.copy()

    prediction = model.predict(img_pre)
    prob = float(prediction[0][0]) if prediction.shape[-1] == 1 else float(np.max(prediction))
    pred_class = "Pneumonia" if prob > 0.55 else "Normal"

    img_uint8 = np.clip(img_array[0], 0, 255).astype(np.uint8)
    return pred_class, prob, img_pre, img_uint8


# 📤 Streamlit File Upload + Grad-CAM
uploaded_file = st.file_uploader("📁 Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    image = Image.open(temp_path).convert("RGB")
    st.image(image, caption="🩻 Uploaded Image", use_container_width=True)

    st.subheader("Model Prediction:")
    pred_class, prob, img_pre, img_uint8 = predict_image(model, image)
    st.write(f"*Prediction:* {pred_class}")
    st.write(f"*Confidence:* {prob:.2%}")

    try:
        temp_gradcam_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        generate_gradcam(model, img_pre, save_path=temp_gradcam_path)

        gradcam_image = Image.open(temp_gradcam_path)
        st.image(gradcam_image, caption="Grad-CAM Visualization", use_container_width=True)

        with open(temp_gradcam_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="gradcam_result.png">📥 Download Grad-CAM Image</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠ Grad-CAM generation failed: {e}")

    finally:
        time.sleep(0.15)
        for p in [temp_gradcam_path, temp_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except PermissionError:
                pass