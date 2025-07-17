import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

DATASET_PATH = "split_data"
MODEL_SAVE_PATH = "mobilenet_model.h5"
PLOT_DIR = "training_plots_mobilenet"
os.makedirs(PLOT_DIR, exist_ok=True)

img_gen = ImageDataGenerator(rescale=1./255)

train_gen = img_gen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_gen = img_gen.flow_from_directory(
    os.path.join(DATASET_PATH, 'val'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)

def plot_metrics(history, folder):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder, 'accuracy.png'))
    plt.clf()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder, 'loss.png'))
    plt.clf()

plot_metrics(history, PLOT_DIR)