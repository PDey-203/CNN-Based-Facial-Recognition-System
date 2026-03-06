import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0

train_ds = keras.utils.image_dataset_from_directory(
    directory=os.path.join(os.path.dirname(__file__), "train"),
    image_size=(128, 128),
    batch_size=64,
    label_mode="int",
    color_mode="rgb",
)

val_ds = keras.utils.image_dataset_from_directory(
    directory=os.path.join(os.path.dirname(__file__), "validation"),
    image_size=(128, 128),
    batch_size=64,
    label_mode="int",
    color_mode="rgb",
)
NUM_CLASSES = len(train_ds.class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


all_labels = np.concatenate([y.numpy() for x, y in train_ds])
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(all_labels), y=all_labels
)
class_weights = dict(enumerate(class_weights))


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ]
)

base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)
base_model.trainable = False

model = Sequential(
    [
        layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks,
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    class_weight=class_weights,
    callbacks=callbacks,
)

save_path = os.path.join(os.path.dirname(__file__), "FER_Model.keras")
model.save(save_path)
print("Model saved at:", save_path)
