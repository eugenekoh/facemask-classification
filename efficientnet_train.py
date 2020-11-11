import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from efficientnet.tfkeras import EfficientNetB0


def constructEfficientNet(config):
    DROPOUT = float(config["dropout"])
    LEARNING_RATE = float(config["learning_rate"])
    IMAGE_SIZE = int(config["image_size"])
    NUM_CLASSES = int(config["num_classes"])

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    x = Dropout(DROPOUT, name="top_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    model = Model(inputs, outputs, name="EfficientNet")

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def unfreezeModel(model, LEARNING_RATE=1e-4):
    # Unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model
