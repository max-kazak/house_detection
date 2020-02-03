# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import cv2

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def process_img(input_filepath, output_filepath, img_size, input_format='.tif'):
    img = cv2.imread(input_filepath)
    img = cv2.resize(img, dsize=(img_size, img_size))
    cv2.imwrite(output_filepath, img)


def process_imgs_masks(input_path, output_path,
                       set_name,
                       img_size,
                       areas, start_ind, end_ind,
                       input_format, output_format):
    logger.info('Processing images for {} dataset'.format(set_name))

    Path(os.path.join(output_path, set_name, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, set_name, 'masks')).mkdir(parents=True, exist_ok=True)

    for area in areas:
        for i in range(start_ind, end_ind + 1):
            filename_in = area + str(i) + input_format
            filename_out = area + str(i) + output_format
            logger.debug('Processing image and mask for {}'.format(filename_in))
            # process image
            process_img(os.path.join(input_path, 'AerialImageDataset', 'images', filename_in),
                        os.path.join(output_path, set_name, 'images', filename_out),
                        img_size, input_format
                        )
            # process mask
            process_img(os.path.join(input_path, 'AerialImageDataset', 'gt', filename_in),
                        os.path.join(output_path, set_name, 'masks', filename_out),
                        img_size, input_format
                        )


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--n_test', default=5, help='Number of samples reserved for test.')
@click.option('--p_val', default=.3, help='Percentage of remaining samples reserved for validation.')
@click.option('--img_size', default=1500, help='Size of the processed image (squared).')
@click.option('--input_format', default='.tif')
@click.option('--output_format', default='.png')
def main(input_path, output_path, n_test, p_val, img_size, input_format, output_format):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        Expects AerialImageDataset/gt and AerialImageDataset/images structure in the input_filepath.
        Use 'make download_date' to follow this structure.
    """
    logger.info('making final data set from {} into {}'.format(input_path, output_path))

    areas = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    max_ind = 36
    n_val = int((max_ind - n_test) * p_val)

    process_imgs_masks(input_path, output_path, 'test', img_size, areas, 1, n_test, input_format, output_format)
    process_imgs_masks(input_path, output_path, 'val', img_size, areas, n_test + 1, n_test + n_val, input_format, output_format)
    process_imgs_masks(input_path, output_path, 'train', img_size, areas, n_test + n_val + 1, max_ind, input_format, output_format)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
