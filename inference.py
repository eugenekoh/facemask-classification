import argparse
import logging
import os

import cv2
import numpy as np
import tensorflow as tf

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def load_model(file):
    model = tf.keras.models.load_model(file)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    logger.info("Successfully loaded model")
    return model


def main(MODEL_FILE, RESIZE_FACTOR, IMAGE_SIZE, OUTPUT_FILE=None):
    model = load_model(MODEL_FILE)
    labels_dict = {0: 'no mask', 1: 'mask'}
    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

    webcam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if OUTPUT_FILE:
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (640, 480))
    classifier = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    while webcam.isOpened():
        (rval, im) = webcam.read()
        im = cv2.flip(im, 1, 1)
        mini = cv2.resize(im, (im.shape[1] // RESIZE_FACTOR, im.shape[0] // RESIZE_FACTOR))
        faces = classifier.detectMultiScale(mini)
        for f in faces:
            (x, y, w, h) = [v * RESIZE_FACTOR for v in f]
            face_img = im[y:y + h, x:x + w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
            face_img = np.reshape(face_img, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
            face_img = face_img / 255.0
            probabilities = model.predict(face_img)
            label = np.argmax(probabilities)
            cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Inference', im)
        if OUTPUT_FILE:
            out.write(im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    if OUTPUT_FILE:
        out.release()
        logger.info(f"Saved video at {OUTPUT_FILE}")
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Saved model directory")
    parser.add_argument("--resize", type=str, default=1, help="Webcam resize factor")
    parser.add_argument("--imgsize", type=str, default=150, help="Image size")
    parser.add_argument("--output", type=str, required=False, help="Output video")
    args = parser.parse_args()
    MODEL_FILE, RESIZE_FACTOR, IMAGE_SIZE = args.model, int(args.resize), int(args.imgsize)

    if args.output == '':
        main(MODEL_FILE, RESIZE_FACTOR, IMAGE_SIZE)
    else:
        OUTPUT_FILE = args.output
        main(MODEL_FILE, RESIZE_FACTOR, IMAGE_SIZE, OUTPUT_FILE)
