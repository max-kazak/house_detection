import logging

import matplotlib.pyplot as plt
from pathlib import Path

from segmentation_models import Unet
from keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import src.models.generators as gen
import src.models.augmentations as aug

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class Network(object):

    def __init__(self,
                 train_data_root_dir='./data/processed/train/',
                 val_data_root_dir='./data/processed/val/',
                 test_data_root_dir='./data/processed/test/',
                 image_folder='images', mask_folder='masks',
                 train_img_size=512,
                 train_batch_size=8,
                 model_out='./models/houses_model_unet.h5'):

        self.train_data_root_dir = train_data_root_dir
        self.val_data_root_dir = val_data_root_dir
        self.test_data_root_dir = test_data_root_dir
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.train_img_size = train_img_size
        self.train_batch_size = train_batch_size
        self.model_out = model_out

        self.model = Unet(backbone_name='efficientnetb0', encoder_weights='imagenet', encoder_freeze=False)
        self.history = None  # generated after train()

    def get_generators(self):
        """
        Return: train_generator, val_generator, test_generator
        """
        train_generator = gen.DataGeneratorFolder(root_dir=self.train_data_root_dir,
                                                  image_folder=self.image_folder,
                                                  mask_folder=self.mask_folder,
                                                  batch_size=self.train_batch_size,
                                                  augmentation=aug.aug_with_crop,
                                                  image_size=self.train_img_size,
                                                  nb_y_features=1)

        val_generator = gen.DataGeneratorFolder(root_dir=self.val_data_root_dir,
                                                image_folder=self.image_folder,
                                                mask_folder=self.mask_folder,
                                                batch_size=1,
                                                nb_y_features=1)

        test_generator = gen.DataGeneratorFolder(root_dir=self.test_data_root_dir,
                                                 image_folder=self.image_folder,
                                                 mask_folder=self.mask_folder,
                                                 batch_size=1,
                                                 nb_y_features=1)

        return train_generator, val_generator, test_generator

    def get_callbacks(self):
        """
        model_autosave: saves best model
        lr_reducer: reduce learning rate when training progress halts
        early_stopping: stop training when validation accuracy stops increasing
        tensorboard: live monitoring of the training process

        Return: callbacks = [model_autosave, lr_reducer, tensorboard, early_stopping]
        """

        # reduces learning rate on plateau
        lr_reducer = ReduceLROnPlateau(factor=0.1,
                                       cooldown=10,
                                       patience=10, verbose=1,
                                       min_lr=0.1e-5)
        # autosave models
        model_autosave = ModelCheckpoint(self.model_out, monitor='val_iou_score',
                                         mode='max', save_best_only=True, verbose=1, period=10)

        # stop learining as metric on validatopn stop increasing
        early_stopping = EarlyStopping(patience=10, verbose=1, mode='auto')

        callbacks = [model_autosave, lr_reducer, early_stopping]

        return callbacks

    def plot_training_history(self, output_dir=None):
        """
        Plots model training history

        Return: plotly fig if output_dir is not specified
        """
        if self.history is not None:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
            ax_loss.plot(self.history.epoch, self.history.history["loss"], label="Train loss")
            ax_loss.plot(self.history.epoch, self.history.history["val_loss"], label="Validation loss")
            ax_loss.legend()
            ax_acc.plot(self.history.epoch, self.history.history["iou_score"], label="Train iou")
            ax_acc.plot(self.history.epoch, self.history.history["val_iou_score"], label="Validation iou")
            ax_acc.legend()
            if output_dir is not None:
                fig.savefig(output_dir)
            else:
                return fig
        else:
            logger.error("can't generate training history plot. No history found. Have you trained the model first?")

    def train(self):
        train_generator, val_generator, _ = self.get_generators()
        callbacks = self.get_callbacks()

        self.model.compile(optimizer=Adam(),
                           loss=bce_jaccard_loss, metrics=[iou_score])

        self.history = self.model.fit_generator(train_generator, shuffle=True,
                                                epochs=50, workers=4, use_multiprocessing=True,
                                                validation_data=val_generator,
                                                verbose=1, callbacks=callbacks)

    def test(self, out=None):
        _, _, test_generator = self.get_generators()

        scores = self.model.evaluate_generator(test_generator)
        metrics = [iou_score]
        report = "Loss: {:.5}\n".format(scores[0])
        for metric, value in zip(metrics, scores[1:]):
            report += "mean {}: {:.5}\n".format(metric.__name__, value)

        logger.info(report)

        if out is not None:
            f = open(out, 'w+')
            f.write(report)
            f.close()


def main():
    net = Network()
    net.train()
    net.plot_training_history(r'./reports/graphs/train_history.png')
    net.test(r'.reports/report.txt')


if __name__ == "__main__":
    main()
