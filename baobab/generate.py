# -*- coding: utf-8 -*-
"""Generating the training data.

This script generates the training data according to the config specifications.

Example
-------
To run this script, pass in the desired config file as argument::

    $ python generate.py configs/tdlmc_diagonal_config.py

"""

import os, sys
import time
import random
import argparse
from types import SimpleNamespace
from pkg_resources import resource_filename
from tqdm import tqdm
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
sys.path.insert(0, '/home/jwp/stage/sl/lenstronomy')
import lenstronomy
# custom config class
from baobab.configs import Config
# BNN prior class
import baobab.bnn_priors as bnn_priors
# Lenstronomy modules
print(lenstronomy.__path__)
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
#from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.util as util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

def parse_args():
    """Parses command-line arguments

    Note: there's currently just one -- the config file.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('n_data', default=None, type=int,
                        help='size of dataset to generate (overrides config file)')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = SimpleNamespace()
        args.config = sys.argv[0]
        args.n_data = sys.argv[1]
    return args

def get_PSF_models(psf_config, pixel_scale):
    """Instantiate PSF models by reading in template PSF maps

    Parameters
    ----------
    psf_config : dict
        copy of the PSF config
    pixel_scale : float
        pixel scale in arcsec/pix

    Returns
    -------
    list
        list of lenstronomy PSF instances
    """ 
    psf_models = []
    if psf_config.type == 'PIXEL':
        if psf_config.which_psf_maps is None:
            # Instantiate PSF with all available PSF maps
            #FIXME: equate psf_id with psf_i since seed number is meaningless
            psf_id_list = [101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118]
        else:
            psf_id_list = [psf_config.which_psf_maps]

        for psf_i, psf_id in enumerate(psf_id_list):
            psf_path = resource_filename('baobab.in_data', 'psf_maps/psf_{:d}.fits'.format(psf_id))
            psf_map = pyfits.getdata(psf_path)
            kernel_cut = kernel_util.cut_psf(psf_map, psf_config.kernel_size)
            kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': pixel_scale, 'kernel_point_source': kernel_cut}
            psf_models.append(PSF(**kwargs_psf))
    else:
        raise NotImplementedError
    return psf_models

def amp_to_mag_extended(mag_kwargs_list, light_model, data_api):
    import copy
    amp_kwargs_list = copy.deepcopy(mag_kwargs_list)
    for i, mag_kwargs in enumerate(mag_kwargs_list):
        amp_kwargs = amp_kwargs_list[i]
        mag = amp_kwargs.pop('magnitude')
        cps_norm = light_model.total_flux(amp_kwargs_list, norm=True, k=i)[0]
        cps = data_api.magnitude2cps(mag)
        amp = cps/ cps_norm
        amp_kwargs['amp'] = amp 
    return amp_kwargs_list

def amp_to_mag_point(mag_kwargs_list, point_source_model, data_api):
    amp_kwargs_list = mag_kwargs_list.copy()
    amp_list = []
    for i, mag_kwargs in enumerate(mag_kwargs_list):
        amp_kwargs = amp_kwargs_list[i]
        mag = np.array(amp_kwargs.pop('magnitude'))
        cps_norm = 1.0
        cps = data_api.magnitude2cps(mag)
        amp = cps/ cps_norm
        amp_list.append(amp)
    amp_kwargs_list = point_source_model.set_amplitudes(amp_list, amp_kwargs_list)
    return amp_kwargs_list

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.n_data is not None:
        cfg.n_data = args.n_data
    # Seed for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)
        print("Destination folder: {:s}".format(cfg.out_dir))
    else:
        raise OSError("Destination folder already exists.")

    # Instantiate PSF models
    psf_models = get_PSF_models(cfg.psf, cfg.instrument.pixel_scale)
    n_psf = len(psf_models)

    # Instantiate ImageData
    #kwargs_data = sim_util.data_configure_simple(**cfg.image)
    #image_data = ImageData(**kwargs_data)

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

    # Initialize dataframe of labels
    param_list = []
    for comp in cfg.components:
        param_list += ['{:s}_{:s}'.format(comp, param) for param in bnn_prior.params[cfg.bnn_omega[comp]['profile']] ]
    if 'agn_light' in cfg.components:
        param_list += ['magnification_{:d}'.format(i) for i in range(4)]
        param_list += ['x_image_{:d}'.format(i) for i in range(4)]
        param_list += ['y_image_{:d}'.format(i) for i in range(4)]
        param_list += ['n_img']
    metadata = pd.DataFrame(columns=param_list)

    print("Starting simulation...")
    for i in tqdm(range(cfg.n_data)):
        psf_model = psf_models[i%n_psf]
        sample = bnn_prior.sample() # FIXME: sampling in batches

        # Instantiate SimAPI (converts mag to amp and wraps around image model)
        kwargs_detector = util.merge_dicts(cfg.instrument, cfg.bandpass, cfg.observation)
        kwargs_detector.update(seeing=cfg.psf.fwhm,
                               psf_type=cfg.psf.type,
                               psf_model=psf_model) # keyword deprecation warning: I asked Simon to change this to kernel_point_source
        data_api = DataAPI(cfg.image.num_pix, **kwargs_detector)
        image_data = data_api.data_class

        #sim_api = SimAPI(numpix=cfg.image.num_pix, 
        #                 kwargs_single_band=kwargs_detector,
        #                 kwargs_model=kwargs_model, 
        #                 kwargs_numerics=cfg.numerics)

        kwargs_lens_mass = [sample['lens_mass'], sample['external_shear']]
        kwargs_src_light = [sample['src_light']]
        kwargs_src_light = amp_to_mag_extended(kwargs_src_light, src_light_model, data_api)
        kwargs_lens_light = None
        kwargs_ps = None

        if 'agn_light' in cfg.components:
            x_image, y_image = lens_eq_solver.findBrightImage(sample['src_light']['center_x'],
                                                              sample['src_light']['center_y'],
                                                              kwargs_lens_mass,
                                                              numImages=4,
                                                              min_distance=cfg.instrument.pixel_scale, 
                                                              search_window=cfg.image.num_pix*cfg.instrument.pixel_scale)
            magnification = lens_mass_model.magnification(x_image, y_image, kwargs=kwargs_lens_mass)
            unlensed_mag = sample['agn_light']['magnitude']
            lensed_mag = np.abs(magnification)*unlensed_mag
            kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image, 'magnitude': lensed_mag}]
            kwargs_ps = amp_to_mag_point(kwargs_ps, ps_model, data_api)

        if 'lens_light' in cfg.components:
            kwargs_lens_light = [sample['lens_light']]
            kwargs_lens_light = amp_to_mag_extended(kwargs_lens_light, lens_light_model, data_api)

        # Instantiate image model
        image_model = ImageModel(image_data, psf_model, lens_mass_model, src_light_model,
                                 lens_light_model, ps_model, kwargs_numerics=cfg.numerics)

        # Generate image
        img = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)

        #kwargs_in_amp = sim_api.magnitude2amplitude(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)
        #imsim_api = sim_api.image_model_class
        #imsim_api.image(*kwargs_in_amp)

        # Add noise
        noise = data_api.noise_for_model(img, background_noise=True, poisson_noise=True, seed=cfg.seed)
        img += noise

        # Save image file
        img_path = os.path.join(cfg.out_dir, 'X_{0:07d}.npy'.format(i+1))
        np.save(img_path, img)

        # Save labels
        meta = {}
        for comp in cfg.components:
            for param_name, param_value in sample[comp].items():
                meta['{:s}_{:s}'.format(comp, param_name)] = param_value
        if 'agn_light' in cfg.components:
            n_img = len(x_image)
            for i in range(n_img):
                meta['magnification_{:d}'.format(i)] = magnification[i]
                meta['x_image_{:d}'.format(i)] = x_image[i]
                meta['y_image_{:d}'.format(i)] = y_image[i]
                meta['n_img'] = n_img
        metadata = metadata.append(meta, ignore_index=True)

    # Fix column ordering
    metadata = metadata[param_list]
    metadata_path = os.path.join(cfg.out_dir, 'metadata.csv')
    metadata.to_csv(metadata_path, index=None)
    print("Labels include: ", metadata.columns.values)

if __name__ == '__main__':
    main()
