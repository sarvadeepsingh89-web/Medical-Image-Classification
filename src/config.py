import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.applications.resnet50 import preprocess_input

# 🔒 Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ⚙️ Dataset Paths
BASE_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# 💾 Model Save Paths
MODEL_DIR = "/kaggle/working/model"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_HEAD_MODEL = os.path.join(MODEL_DIR, "best_resnet_head.keras")
BEST_FINETUNE_MODEL = os.path.join(MODEL_DIR, "best_resnet_finetune.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_resnet_pneumonia_model.keras")
CSV_LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")

# 🖼️ Image Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 🧮 Training Parameters
EPOCHS = 20
LEARNING_RATE = 1e-4
L2_REG = 0.001

# 🧩 Register Serializable Preprocess Function
@tf.keras.utils.register_keras_serializable()
def _preprocess_input(x):
    return tf.keras.applications.resnet50.preprocess_input(x)

