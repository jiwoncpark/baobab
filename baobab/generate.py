# -*- coding: utf-8 -*-
"""Generating the training data.

This script generates the training data according to the config specifications.

Example
-------
To run this script, pass in the desired config file as argument::

    $ generate baobab/configs/tdlmc_diagonal_config.py --n_data 1000

"""

import os, sys
import random
import argparse
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import pandas as pd

# Lenstronomy modules
import lenstronomy
print("Lenstronomy path being used: {:s}".format(lenstronomy.__path__[0]))
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.util as util
# Baobab modules
from baobab.configs import BaobabConfig
import baobab.bnn_priors as bnn_priors
from baobab.sim_utils import get_PSF_models, generate_image

def parse_args():
    """Parse command-line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--n_data', default=None, dest='n_data', type=int,
                        help='size of dataset to generate (overrides config file)')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = SimpleNamespace()
        args.config = sys.argv[0]
        args.n_data = sys.argv[1]
    return args

def main():
    args = parse_args()
    cfg = BaobabConfig.from_file(args.config)
    if args.n_data is not None:
        cfg.n_data = args.n_data
    # Seed for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # Create data directory
    save_dir = cfg.out_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Destination folder: {:s}".format(save_dir))
    else:
        raise OSError("Destination folder already exists.")
    # Instantiate PSF models
    psf_models = get_PSF_models(cfg.psf, cfg.instrument.pixel_scale)
    n_psf = len(psf_models)
    # Instantiate density models
    kwargs_model = dict(
                    lens_model_list=[cfg.bnn_omega.lens_mass.profile, cfg.bnn_omega.external_shear.profile],
                    source_light_model_list=[cfg.bnn_omega.src_light.profile],
                    )       
    lens_mass_model = LensModel(lens_model_list=kwargs_model['lens_model_list'])
    src_light_model = LightModel(light_model_list=kwargs_model['source_light_model_list'])
    lens_eq_solver = LensEquationSolver(lens_mass_model)
    lens_light_model = None
    ps_model = None                                     
    if 'lens_light' in cfg.components:
        kwargs_model['lens_light_model_list'] = [cfg.bnn_omega.lens_light.profile]
        lens_light_model = LightModel(light_model_list=kwargs_model['lens_light_model_list'])
    if 'agn_light' in cfg.components:
        kwargs_model['point_source_model_list'] = [cfg.bnn_omega.agn_light.profile]
        ps_model = PointSource(point_source_type_list=kwargs_model['point_source_model_list'], fixed_magnification_list=[False])
    # Initialize BNN prior
    bnn_prior = getattr(bnn_priors, cfg.bnn_prior_class)(cfg.bnn_omega, cfg.components)
    # Initialize empty metadata dataframe
    metadata = pd.DataFrame()
    current_idx = 0 # running idx of dataset
    pbar = tqdm(total=cfg.n_data)
    while current_idx < cfg.n_data:
        sample = bnn_prior.sample() # FIXME: sampling in batches
        # Selections (except selection on total magnification, which comes later)
        #TODO: this is getting longer so separate them
        if sample['lens_mass']['theta_E'] < cfg.selection.theta_E.min:
            continue
        lens_mass_e = (sample['lens_mass']['e1']**2.0 + sample['lens_mass']['e2']**2.0)**0.5
        if lens_mass_e > 1.0:
            continue
        src_light_e = (sample['src_light']['e1']**2.0 + sample['src_light']['e2']**2.0)**0.5
        if src_light_e > 1.0:
            continue
        if 'lens_light' in cfg.components:
            lens_light_e = (sample['lens_light']['e1']**2.0 + sample['lens_light']['e2']**2.0)**0.5
            if lens_light_e > 1.0:
                continue
        psf_model = psf_models[current_idx%n_psf]
        # Detector and observation conditions
        kwargs_detector = util.merge_dicts(cfg.instrument, cfg.bandpass, cfg.observation)
        kwargs_detector.update(seeing=cfg.psf.fwhm,
                               psf_type=cfg.psf.type,
                               kernel_point_source=psf_model)
        data_api = DataAPI(cfg.image.num_pix, **kwargs_detector)
        # Generate the image
        img, img_features = generate_image(sample, psf_model, data_api, lens_mass_model, src_light_model, lens_eq_solver, cfg.instrument.pixel_scale, cfg.image.num_pix, cfg.components, cfg.numerics, min_magnification=cfg.selection.magnification.min, lens_light_model=lens_light_model, ps_model=ps_model)
        if img is None: # couldn't make the magnification cut
            continue
        # Save image file
        img_filename = 'X_{0:07d}.npy'.format(current_idx)
        img_path = os.path.join(save_dir, img_filename)
        np.save(img_path, img)
        # Save labels
        meta = {}
        for comp in cfg.components:
            for param_name, param_value in sample[comp].items():
                meta['{:s}_{:s}'.format(comp, param_name)] = param_value
        if 'agn_light' in cfg.components:
            n_img = len(img_features['x_image'])
            for i in range(n_img):
                meta['x_image_{:d}'.format(i)] = img_features['x_image'][i]
                meta['y_image_{:d}'.format(i)] = img_features['y_image'][i]
                meta['n_img'] = n_img
        if cfg.bnn_prior_class in ['EmpiricalBNNPrior', 'DiagonalCosmoBNNPrior']:
            for misc_name, misc_value in sample['misc'].items():
                meta['{:s}'.format(misc_name)] = misc_value
        meta['total_magnification'] = img_features['total_magnification']
        meta['img_filename'] = img_filename
        metadata = metadata.append(meta, ignore_index=True)
        # Update progress
        current_idx += 1
        pbar.update(1)
    pbar.close()

    # Store source pos offset from lens center, which is what we draw
    metadata['src_light_pos_offset_x'] = metadata['src_light_center_x'] - metadata['lens_mass_center_x']
    metadata['src_light_pos_offset_y'] = metadata['src_light_center_y'] - metadata['lens_mass_center_y']

    # Sort columns lexicographically
    metadata = metadata.reindex(sorted(metadata.columns), axis=1)
    metadata_path = os.path.join(save_dir, 'metadata.csv')
    metadata.to_csv(metadata_path, index=None)
    print("Labels include: ", metadata.columns.values)

if __name__ == '__main__':
    main()
