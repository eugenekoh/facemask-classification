import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import load_img, array_to_img
from matplotlib import cm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model

# Constants specific to model
IMAGE_SIZE = 150
LABELS_DICT = {0: 'no mask', 1: 'mask'}


def main(model_path, images):
    """
    This Grad-CAM visualization code is specific to EfficientNetB0 architecture.
    Modified from https://keras.io/examples/vision/grad_cam/
    """

    # Load model
    model = load_model(model_path)
    last_conv_layer_name = "top_activation"  # change this if using another model
    classifier_layer_names = [
        "avg_pool",
        "batch_normalization",
        "top_dropout",
        "pred"
    ]

    # inputs to last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # last conv layer to predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    for image_path in images:
        # Load image
        image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Calculate gradients
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)

            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)

        # Superimpose heatmap
        img = load_img(image_path)
        img = img_to_array(img)

        jet = cm.get_cmap("viridis")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.6 + img
        superimposed_img = array_to_img(superimposed_img)

        new_name = Path(image_path).stem + '_grad_cam.jpg'
        save_path = Path(image_path).parent / new_name

        plt.imshow(superimposed_img)
        plt.axis('off')

        label = LABELS_DICT[int(top_pred_index)]
        confidence = float(preds[0][top_pred_index])
        plt.title(f"{label}: {confidence:.2f}")
        plt.savefig(save_path, bbox_inches='tight')

        print(f"saved grad-cam visualization in {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='./models/efficientnet', help="Saved model directory")
    parser.add_argument("--img", type=str, default='./assets/images',
                        help="Image file path or directory of JPG images")

    args = parser.parse_args()

    images = Path(args.img)
    if images.is_dir():
        images = list(images.glob('*.jpg'))
    else:
        images = [images]
    main(args.model, images)
