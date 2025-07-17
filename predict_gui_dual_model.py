import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

# === Cấu hình ===
MODEL_CNN_PATH = "cnn_fruit_model.h5"
MODEL_MOBILENET_PATH = "mobilenet_model.h5"
DATASET_DIR = "Unified_Dataset"
IMG_SIZE = (128, 128)

# === Load mô hình ===
model_cnn = load_model(MODEL_CNN_PATH)
model_mobilenet = load_model(MODEL_MOBILENET_PATH)

# === Lấy class name ===
CLASS_NAMES = sorted([
    f"{fruit}_{state}"
    for fruit in sorted(os.listdir(DATASET_DIR))
    for state in sorted(os.listdir(os.path.join(DATASET_DIR, fruit)))
])

# === Đổi nhãn sang tiếng Việt ===
def convert_label(label):
    label = label.replace("fresh", "tươi")
    label = label.replace("rotten", "hỏng")
    label = label.replace("_", " - ")
    return label

# === Dự đoán ảnh và trả về nhãn + độ tin cậy ===
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    pred = model.predict(img_tensor, verbose=0)[0]
    predicted_idx = np.argmax(pred)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = pred[predicted_idx] * 100
    return convert_label(predicted_class), confidence

# === Xử lý khi chọn ảnh ===
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    # Hiển thị ảnh
    img = Image.open(file_path)
    img = img.resize((350, 350))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Dự đoán
    label_cnn, conf_cnn = predict_image(file_path, model_cnn)
    label_mobilenet, conf_mobilenet = predict_image(file_path, model_mobilenet)

    result_label_cnn.config(text=f"[CNN] {label_cnn} ({conf_cnn:.1f}%)")
    result_label_mobilenet.config(text=f"[MobileNetV2] {label_mobilenet} ({conf_mobilenet:.1f}%)")


# === Giao diện chính ===
root = tk.Tk()
root.title("Dự đoán tình trạng thực phẩm với 2 mô hình")
root.geometry("550x550")

# Căn giữa
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (550 / 2))
y = int((screen_height / 2) - (550 / 2))
root.geometry(f"+{x}+{y}")

btn = tk.Button(root, text="Chọn ảnh để dự đoán", command=open_file, font=("Arial", 14))
btn.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label_cnn = tk.Label(root, text="[CNN] Chưa có dự đoán", font=("Arial", 12))
result_label_cnn.pack(pady=5)

result_label_mobilenet = tk.Label(root, text="[MobileNetV2] Chưa có dự đoán", font=("Arial", 12))
result_label_mobilenet.pack(pady=5)

root.mainloop()
