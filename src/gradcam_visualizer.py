import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, test_images, test_labels, index=3, save_path="GradCAM_Final.png"):

    # 1️⃣ Pick a test image
    img_tensor = test_images[index]
    true_label = int(test_labels[index].numpy().squeeze())

    # 2️⃣ Predict class
    pred = model.predict(tf.expand_dims(img_tensor, axis=0))
    prob = float(pred[0][0]) if pred.shape[-1] == 1 else float(np.max(pred))
    pred_class = int(prob > 0.55) if pred.shape[-1] == 1 else np.argmax(pred)
    print(f"✅ True label: {true_label}, Predicted class: {pred_class}, Probability: {prob:.4f}")

    # 3️⃣ Convert image for visualization
    img_np = img_tensor.numpy()
    img_uint8 = (img_np * 255).astype(np.uint8) if img_np.max() <= 1 else np.clip(img_np, 0, 255).astype(np.uint8)

    # 4️⃣ Extract the ResNet base (submodel)
    resnet_base = None
    for layer in model.layers:
        if "resnet" in layer.name.lower():
            resnet_base = layer
            break
    if resnet_base is None:
        raise ValueError("❌ Could not find ResNet backbone in model")

    # 5️⃣ Find last convolutional layer
    last_conv_layer = None
    for layer in reversed(resnet_base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        raise ValueError("❌ Could not find Conv2D layer in ResNet50")
    print(f"✅ Using last conv layer: {last_conv_layer}")

    # 6️⃣ Build grad model from *resnet_base directly*
    conv_layer = resnet_base.get_layer(last_conv_layer)
    grad_model = tf.keras.models.Model(
        inputs=resnet_base.input,
        outputs=[conv_layer.output, resnet_base.output]
    )

    # 7️⃣ Preprocess image same as model
    img_input = tf.expand_dims(img_tensor, axis=0)
    preprocess_layer = model.get_layer("preprocess")  
    img_pre = preprocess_layer(img_input)             

    # 8️⃣ Forward + Gradient
    with tf.GradientTape() as tape:
        conv_outputs, base_features = grad_model(img_pre, training=False)
        x = model.get_layer("gap")(base_features)
        for layer_name in ["dropout", "dense", "batch_normalization", "dropout_1", "output"]:
            if layer_name in [l.name for l in model.layers]:
                x = model.get_layer(layer_name)(x)
        preds = x
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 9️⃣ Build heatmap
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # 🔟 Overlay heatmap on image
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)

    # 1️⃣1️⃣ Plot and save
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_uint8)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM | Pred: {pred_class} | True: {true_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print("✅ Grad-CAM saved to:", os.path.abspath(save_path))
