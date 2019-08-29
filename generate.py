import os, sys
import time
import random
from tqdm import tqdm
import astropy.io.fits as pyfits
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# custom config class
from configs.parser import Config
# BNN prior class
import bnn_priors
# Lenstronomy modules
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)
        print("Destination folder: {:s}".format(cfg.out_dir))
    else:
        raise OSError("Destination folder already exists.")

    # Instantiate PSF with available PSF maps
    psf_seed_list = [101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118]
    n_psf = len(psf_seed_list)
    psf_models = []
    for psf_i, psf_seed in enumerate(psf_seed_list):
        psf_path = os.path.join(cfg.psf.kernel_dir, 'psf_{:d}.fits'.format(psf_seed))
        psf_map = pyfits.getdata(psf_path)
        kernel_cut = kernel_util.cut_psf(psf_map, cfg.psf.kernel_size)
        kwargs_psf = {'psf_type': cfg.psf.type, 'pixel_size': cfg.image.deltaPix, 'kernel_point_source': kernel_cut}
        psf_models.append(PSF(**kwargs_psf))

    # Instantiate ImageData
    kwargs_data = sim_util.data_configure_simple(**cfg.image)
    image_data = ImageData(**kwargs_data)

    # Instantiate models
    lens_mass_model = LensModel(lens_model_list=cfg.bnn_omega.lens_mass.type)
    src_light_model = LightModel(light_model_list=cfg.bnn_omega.src_light.type)
    lens_eq_solver = LensEquationSolver(lens_mass_model)
    lens_light_model = None
    ps_model = None

    if 'lens_light' in cfg.components:
        lens_light_model = LightModel(light_model_list=cfg.bnn_omega.lens_light.type)
    if 'agn_light' in cfg.components:
        ps_model = PointSource(point_source_type_list=cfg.bnn_omega.agn_light.type, fixed_magnification_list=[False])

    # Initialize BNN prior
    bnn_prior = getattr(bnn_priors, cfg.bnn_prior_class)(cfg.bnn_omega, cfg.components)

    # Initialize dataframe of labels
    param_list = []
    for comp in cfg.components:
        param_list += ['{:s}_{:s}'.format(comp, param) for param in cfg['bnn_omega'][comp]['params'] ]
    metadata = pd.DataFrame(columns=param_list)

    print("Starting simulation...")
    for i in tqdm(range(cfg.n_data)):
        psf_model = psf_models[i%n_psf]
        sample = bnn_prior.sample() # FIXME: sampling in batches

        kwargs_lens_mass = [sample['spemd'], sample['ext_shear']]
        kwargs_src_light = [sample['src_sersic']]
        kwargs_lens_light = None
        kwargs_ps = None

        if 'agn_light' in cfg.components:
            x_image, y_image = lens_eq_solver.findBrightImage(sample['src_sersic']['center_x'],
                                                              sample['src_sersic']['center_y'],
                                                              kwargs_lens_mass,
                                                              numImages=4,
                                                              min_distance=cfg.image.deltaPix, 
                                                              search_window=cfg.image.numPix*cfg.image.deltaPix)
            mag = lens_mass_model.magnification(x_image, y_image, kwargs=kwargs_lens_mass)
            unlensed_amp = sample['agn_ps']['amp']
            kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image, 'point_amp': np.abs(mag)*unlensed_amp}]
            
        if 'lens_light' in cfg.components:
            kwargs_lens_light = [sample['lens_sersic']]

        # Instantiate image model
        kwargs_numerics = {'supersampling_factor': 1}
        image_model = ImageModel(image_data, psf_model, lens_mass_model, src_light_model,
                                 lens_light_model, ps_model, kwargs_numerics=kwargs_numerics)
        # Generate image
        img = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)

        # Add noise
        poisson = image_util.add_poisson(img, exp_time=cfg.image.exposure_time)
        bkg = image_util.add_background(img, sigma_bkd=cfg.image.sigma_bkg)
        #img = img + bkg + poisson
        #image_data.update_data(img)

        # Save image file
        img_path = os.path.join(cfg.out_dir, 'X_{0:07d}.npy'.format(i+1))
        np.save(img_path, img)

        #img = Image.fromarray(img, mode='L')
        #img = img.convert('L')
        #img.save(img_path)

        # Save labels
        # Hardcoded but is meant to facilitate introduction of other profiles, e.g. NFW + Sersic...
        meta = {'lens_mass_{:s}'.format(k): v for k, v in sample['spemd'].items()}
        meta.update({'lens_mass_{:s}'.format(k): v for k, v in sample['ext_shear'].items()})
        meta.update({'src_light_{:s}'.format(k): v for k, v in sample['src_sersic'].items()})
        if 'lens_light' in cfg.components:
            meta.update({'lens_light_{:s}'.format(k): v for k, v in sample['lens_sersic'].items()})
        if 'agn_light' in cfg.components:
            meta.update({'agn_light_{:s}'.format(k): v for k, v in sample['agn_ps'].items()})
        metadata = metadata.append(meta, ignore_index=True)

    # Fix column ordering
    metadata = metadata[param_list]
    metadata_path = os.path.join(cfg.out_dir, 'metadata.csv')
    metadata.to_csv(metadata_path, index=None)
    print("Labels include: ", metadata.columns.values)
