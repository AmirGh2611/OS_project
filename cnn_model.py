from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input, \
    GlobalAveragePooling2D

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(256, activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])
