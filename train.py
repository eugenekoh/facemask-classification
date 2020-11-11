import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import json
import pandas as pd
import argparse
import time

from efficientnet_train import constructEfficientNet, unfreezeModel
from utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def constructModel(config, IMAGE_SIZE):
    CONV_1_CHANNEL = int(config['conv_1_channel'])
    CONV_1_WINDOW = int(config["conv_1_window"])
    POOL_1_WINDOW = int(config["pool_1_window"])
    POOL_1_STRIDE = int(config["pool_1_stride"])
    CONV_2_CHANNEL = int(config['conv_2_channel'])
    CONV_2_WINDOW = int(config["conv_2_window"])
    POOL_2_WINDOW = int(config["pool_2_window"])
    POOL_2_STRIDE = int(config["pool_2_stride"])
    DROPOUT = float(config["dropout"])
    OPTIMIZER = config["optimizer"]
    LEARNING_RATE = float(config["learning_rate"])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(CONV_1_CHANNEL, CONV_1_WINDOW, activation='relu',
                               input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(POOL_1_WINDOW, POOL_1_STRIDE),
        tf.keras.layers.Conv2D(CONV_2_CHANNEL, CONV_2_WINDOW, activation='relu'),
        tf.keras.layers.MaxPooling2D(POOL_2_WINDOW, POOL_2_STRIDE),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    if OPTIMIZER == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'Momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.1)
    elif OPTIMIZER == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    else:
        raise Exception("Invalid Optimizer")

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model_name = f"{CONV_1_CHANNEL}_{CONV_2_CHANNEL}_{OPTIMIZER}_{int(DROPOUT * 100)}"

    return model, model_name


def train(model, model_name, config, IMAGE_SIZE):
    TRAINING_DIR = config["train_dir"]
    VALIDATION_DIR = config["val_dir"]
    MODEL_SAVE_DIR = config["model_save_dir"]
    ASSETS_SAVE_DIR = config["assets_save_dir"]
    RESULTS_FILE_PATH = config["results_file_path"]
    EPOCHS = int(config["epochs"])
    BATCH_SIZE = int(config["batch_size"])

    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=BATCH_SIZE,
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE))
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=BATCH_SIZE,
                                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE))

    checkpoint = ModelCheckpoint(f'{MODEL_SAVE_DIR}/{model_name}', monitor='val_loss', verbose=0, save_best_only=True,
                                 mode='auto')
    start_time = time.time()
    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        callbacks=[checkpoint])
    end_time = time.time()
    avg_time = (end_time - start_time) / EPOCHS
    train_loss, val_loss = history.history['loss'], history.history['val_loss']
    train_accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
    get_plots(train_loss, val_loss, "loss", f"{ASSETS_SAVE_DIR}/{model_name}_loss.jpg")
    get_plots(train_accuracy, val_accuracy, "accuracy", f"{ASSETS_SAVE_DIR}/{model_name}_accuracy.jpg")

    final_val_loss, final_val_acc = model.evaluate(validation_generator, batch_size=128)
    if os.path.exists(RESULTS_FILE_PATH):
        results = pd.read_csv(RESULTS_FILE_PATH)
        results = results.append(pd.DataFrame([[model_name, final_val_loss, final_val_acc, avg_time]],
                                              columns=["model_name", "final_val_acc", "final_val_loss", "avg_time"]))
    else:
        results = pd.DataFrame([[model_name, final_val_loss, final_val_acc, avg_time]],
                               columns=["model_name", "final_val_acc", "final_val_loss", "avg_time"])
    results.to_csv(RESULTS_FILE_PATH, index=False)
    print(f"updated results at {RESULTS_FILE_PATH}")


def efficientnet_main(config):
    model_config = config["model"]
    train_config = config["train"]
    IMAGE_SIZE = int(config["model"]["image_size"])

    # train top layer
    model = constructEfficientNet(model_config)
    train(model, "top", train_config, IMAGE_SIZE)

    # train top layer
    model = unfreezeModel(model)
    train(model, "unfreeze", train_config, IMAGE_SIZE)


def main(config):
    model_config = config["model"]
    train_config = config["train"]
    IMAGE_SIZE = int(config["augment"]["image_size"])
    model, model_name = constructModel(model_config, IMAGE_SIZE)
    train(model, model_name, train_config, IMAGE_SIZE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--efficientnet", action="store_true", help="Train efficientnet flag")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    if args.efficientnet:
        efficientnet_main(config)
    else:
        main(config)
