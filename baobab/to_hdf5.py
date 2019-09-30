# -*- coding: utf-8 -*-
"""Converting .npy image files and metadata into HDF5

This script converts the baobab data into the HDF5 format.

Example
-------
To run this script, pass in the baobab out_dir path as the first argument and the framework format as the second, e.g.::

    $ to_hdf5 out_data/tdlmc_train_EmpiricalBNNPrior_seed1113 --format 'tf'

The output file will be named `tdlmc_train_EmpiricalBNNPrior_seed1113.h5` and can be found inside the directory provided as the first argument.

See the demo notebook `demo/Read_hdf5_file.ipynb` for instructions on how to access the datasets in this file.

"""

import os, sys
import numpy as np
import pandas as pd
import argparse
import h5py
from tqdm import tqdm

def parse_args():
    """Parses command-line arguments

    """
    parser = argparse.ArgumentParser()
   
    parser.add_argument('npy_dir', 
                        help='directory containing .npy files and metadata (path of out_dir in the baobab config)')
    parser.add_argument('--format', 
                        default='tf', 
                        dest='format', 
                        type=str,
                        choices=['tf', 'theano'],
                        help='format of image. Default: tf.')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = SimpleNamespace()
        args.npy_dir = sys.argv[0]
        args.format = sys.argv[1]

    #base, ext = os.path.splitext(save_path)
    #if ext.lower() not in ['.h5', '.hdf5']:
    #    raise argparse.ArgumentTypeError('out_filepath must have a valid HDF5 extension.')
    return args

def main():
    args = parse_args()
    baobab_out_dir = os.path.basename(os.path.normpath(args.npy_dir))
    save_path = os.path.join(args.npy_dir, '{:s}.h5'.format(baobab_out_dir))
    print("Destination path: {:s}".format(save_path))

    metadata_path = os.path.join(args.npy_dir, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path, index_col=None)

    img_path_list = metadata_df['img_filename'].values
    first_img_filepath = os.path.join(args.npy_dir, img_path_list[0])
    n_x, n_y = np.load(first_img_filepath).shape # image dimensions
    n_data, n_cols = metadata_df.shape

    # Initialize hdf5 file
    hdf_file = h5py.File(save_path, mode='w', driver=None)

    # Create dataset for images
    if args.format == 'tf':
        img_shape = (n_x, n_y, 1) # tf data shape
    elif args.format == 'theano':
        img_shape = (1, n_x, n_y) # theano data shape
    else:
        raise NotImplementedError
    
    # Initialize mean and std of images, and quantities required to compute them online
    hdf_file.create_dataset('pixels_mean', img_shape, np.float32)
    hdf_file.create_dataset('pixels_std', img_shape, np.float32)
    mean = np.zeros(img_shape, np.float32)
    std = np.zeros(img_shape, np.float32)
    sum_sq = np.zeros(img_shape, np.float32) 
    ddof = 0 # degree of freedom

    print("Saving images...")
    current_idx = 0 # running idx of dataset
    pbar = tqdm(total=n_data)
    while current_idx < n_data:

        # Read in image
        img_path = os.path.join(args.npy_dir, img_path_list[current_idx])
        img = np.load(img_path).reshape(img_shape)
        # Change axis order for theano
        if format=='theano':
            img = np.rollaxis(img, 2)

        # Populate images dataset
        dataset_name = 'image_{:d}'.format(current_idx)
        hdf_file.create_dataset(dataset_name, img_shape, np.float32)
        hdf_file[dataset_name][...] = img[None]

        # Update running mean and std (Welford's algorithm)
        current_idx += 1        
        delta = img - mean
        mean += delta / current_idx
        sum_sq += delta * (img - mean)

        # Update progress
        pbar.update(1)
    pbar.close()
    # Populate mean, std datasets
    std = np.sqrt(sum_sq / (n_data - ddof))
    hdf_file['pixels_mean'][...] = mean
    hdf_file['pixels_std'][...] = std
    hdf_file.close()

    # Create dataset for metadata df
    metadata_df.to_hdf(save_path, key='metadata', mode='a', format='table')
    # TODO: serialize or subgroup each row so the whole dataframe isn't read into memory

if __name__ == '__main__':
    main()