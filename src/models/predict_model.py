import os
import logging

import matplotlib.pyplot as plt
import click
from pathlib import Path
import cv2
import numpy as np

from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score
from keras.models import load_model

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

MASK_THRESHOLD = 0.4


def read_img_mask(root_dir, image_folder, mask_folder, img_name, img_size=1472):
    img = cv2.imread(os.path.join(root_dir, image_folder, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img = img / 255

    if mask_folder is not None:
        mask = cv2.imread(os.path.join(root_dir, mask_folder, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_size, img_size))
        mask = (mask > 0).astype(np.int8)
    else:
        mask = None

    return img, mask


def predict(model, img):
    X = np.expand_dims(img, axis=0)
    pred = model.predict(X)
    pred_mask = pred[0].reshape(pred.shape[1], pred.shape[2])
    pred_mask = (pred_mask > MASK_THRESHOLD).astype(np.int8)
    return pred_mask


def gen_report(image_name, out_dir, img, pred_mask, act_mask=None):
    if act_mask is not None:
        f, ax = plt.subplots(1, 3, figsize=(10, 4))
    else:
        f, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img)
    ax[1].imshow(pred_mask, cmap='gray')
    ax[0].set_title('Original ({})'.format(image_name))
    ax[1].set_title('Predicted mask')
    if act_mask is not None:
        ax[2].imshow(act_mask, cmap='gray')
        ax[2].set_title('True mask')

    f.savefig(os.path.join(out_dir, image_name))
    plt.close(f)


@click.command()
@click.argument('images', nargs=-1)
@click.option('--model_path', '-m', type=click.Path(exists=True), help='Path to .h5 model')
@click.option('--root_dir', type=click.Path(exists=True), help='Path to dataset')
@click.option('--image_folder', default='images', help='images folder')
@click.option('--mask_folder', help='(optional) true masks folder for comparisons in reports')
@click.option('--predict_dir', type=click.Path(), help='output dir for predicted masks')
@click.option('--report_dir', type=click.Path(), help='output dir for generated reports')
@click.option('--img_size', default=1472, help='resize images to this size')
def main(images, model_path, root_dir, image_folder, mask_folder, predict_dir, report_dir, img_size):
    logger.info('running predict and reporting on {} dataset'.format(root_dir))

    model = load_model(model_path, custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
                                                   'iou_score': iou_score})

    if len(images) == 0:
        # process all images in image_folder
        images = os.listdir(os.path.join(root_dir, image_folder))

    for image_name in images:
        logger.info('predicting image {}'.format(image_name))
        img, act_mask = read_img_mask(root_dir, image_folder, mask_folder, image_name, img_size)
        pred_mask = predict(model, img)
        if predict_dir is not None:
            Path(predict_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(predict_dir, image_name), pred_mask*255)
        if report_dir is not None:
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            gen_report(image_name, report_dir, img, pred_mask, act_mask)


if __name__ == '__main__':
    main()
