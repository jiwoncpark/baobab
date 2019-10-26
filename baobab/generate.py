# -*- coding: utf-8 -*-
"""Generating the training data.

This script generates the training data according to the config specifications.

Example
-------
To run this script, pass in the desired config file as argument::

    $ generate baobab/configs/tdlmc_diagonal_config.py --n_data 1000

"""

import os, sys
import time
import random
import argparse
import copy
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
print("Lenstronomy path being used: {:s}".format(lenstronomy.__path__[0]))
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
    """Convert the magnitude entries into amp (counts per second)
    used by lenstronomy to render the image, for extended objects

    Parameters
    ----------
    mag_kwargs_list : list
        list of kwargs dictionaries in which 'amp' keys are replaced by 'magnitude'
    light_model : lenstronomy.LightModel object
        light model describing the surface brightness profile, used for calculating
        the total flux. Note that only some profiles with an analytic integral can be
        used.
    data_api : lenstronomy.DataAPI object
        a wrapper object around lenstronomy.Observation that has the magnitude zeropoint
        information, with which the magnitude-to-amp conversion is done.

    Returns
    -------
    list
        list of kwargs dictionaries with 'magnitude' replaced by 'amp'

    """
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
    """Convert the magnitude entries into amp (counts per second)
    used by lenstronomy to render the image, for point sources

    See the docstring for `amp_to_mag_extended` for parameter descriptions.

    """
    amp_kwargs_list = copy.deepcopy(mag_kwargs_list)
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

def get_unlensed_total_flux(kwargs_src_light_list, src_light_model, kwargs_ps_list=None, ps_model=None):
    """Compute the total flux of unlensed objects

    Parameter
    ---------
    kwargs_src_light_list : list
        list of kwargs dictionaries for the unlensed source galaxy, each with an 'amp' key
    kwargs_ps_list : list
        list of kwargs dictionaries for the unlensed point source (if any), each with an 'amp' key

    Returns
    -------
    float
        the total unlensed flux

    """
    total_flux = 0.0
    for i, kwargs_src in enumerate(kwargs_src_light_list):
        total_flux += src_light_model.total_flux(kwargs_src_light_list, norm=True, k=i)[0]
    if kwargs_ps_list is not None:
        assert ps_model is not None
        for i, kwargs_ps in enumerate(kwargs_ps_list):
            total_flux += kwargs_ps['point_amp']
    return total_flux
        
def get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps, image_model):
    """Compute the total flux of the lensed image

    Returns
    -------
    float
        the total lensed flux

    """

    lensed_src_image = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps, lens_light_add=False)
    lensed_total_flux = np.sum(lensed_src_image)
    return lensed_total_flux

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.n_data is not None:
        cfg.n_data = args.n_data
    # Seed for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    save_dir = os.path.join('out_data', cfg.out_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Destination folder: {:s}".format(save_dir))
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
    param_list += ['img_filename', 'total_magnification', 'src_light_pos_offset_x', 'src_light_pos_offset_y']
    if cfg.bnn_prior_class == 'EmpiricalBNNPrior':
        param_list += ['z_lens', 'z_src', 'vel_disp_iso', 'R_eff_lens', 'R_eff_src', 'abmag_src']
    metadata = pd.DataFrame(columns=param_list)

    print("Starting simulation...")
    current_idx = 0 # running idx of dataset
    pbar = tqdm(total=cfg.n_data)
    while current_idx < cfg.n_data:
        psf_model = psf_models[current_idx%n_psf]
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

        # Instantiate SimAPI (converts mag to amp and wraps around image model)
        kwargs_detector = util.merge_dicts(cfg.instrument, cfg.bandpass, cfg.observation)
        kwargs_detector.update(seeing=cfg.psf.fwhm,
                               psf_type=cfg.psf.type,
                               kernel_point_source=psf_model) # keyword deprecation warning: I asked Simon to change this to 
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
            magnification = np.abs(lens_mass_model.magnification(x_image, y_image, kwargs=kwargs_lens_mass))
            unlensed_mag = sample['agn_light']['magnitude'] # unlensed agn mag
            kwargs_unlensed_mag_ps = [{'ra_image': x_image, 'dec_image': y_image, 'magnitude': unlensed_mag}] # note unlensed magnitude
            kwargs_unlensed_amp_ps = amp_to_mag_point(kwargs_unlensed_mag_ps, ps_model, data_api) # note unlensed amp
            kwargs_ps = copy.deepcopy(kwargs_unlensed_amp_ps)
            for kw in kwargs_ps:
                kw.update(point_amp=kw['point_amp']*magnification)
        else:
            kwargs_unlensed_amp_ps = None

        if 'lens_light' in cfg.components:
            kwargs_lens_light = [sample['lens_light']]
            kwargs_lens_light = amp_to_mag_extended(kwargs_lens_light, lens_light_model, data_api)

        # Instantiate image model
        image_model = ImageModel(image_data, psf_model, lens_mass_model, src_light_model,
                                 lens_light_model, ps_model, kwargs_numerics=cfg.numerics)

        # Compute magnification
        lensed_total_flux = get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps, image_model)
        unlensed_total_flux = get_unlensed_total_flux(kwargs_src_light, src_light_model, kwargs_unlensed_amp_ps, ps_model)
        total_magnification = lensed_total_flux/unlensed_total_flux

        # Apply magnification cut
        if total_magnification < cfg.selection.magnification.min:
            continue

        # Generate image for export
        img = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)
        #kwargs_in_amp = sim_api.magnitude2amplitude(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)
        #imsim_api = sim_api.image_model_class
        #imsim_api.image(*kwargs_in_amp)

        # Add noise
        noise = data_api.noise_for_model(img, background_noise=True, poisson_noise=True, seed=None)
        img += noise

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
            n_img = len(x_image)
            for i in range(n_img):
                meta['magnification_{:d}'.format(i)] = magnification[i]
                meta['x_image_{:d}'.format(i)] = x_image[i]
                meta['y_image_{:d}'.format(i)] = y_image[i]
                meta['n_img'] = n_img
        if cfg.bnn_prior_class == 'EmpiricalBNNPrior':
            for misc_name, misc_value in sample['misc'].items():
                meta['{:s}'.format(misc_name)] = misc_value
        meta['total_magnification'] = total_magnification
        meta['img_filename'] = img_filename
        # Store source pos offset from lens center, which is what we draw
        meta['src_light_pos_offset_x'] = meta['src_light_center_x'] - meta['lens_mass_center_x']
        meta['src_light_pos_offset_y'] = meta['src_light_center_y'] - meta['lens_mass_center_y']
        metadata = metadata.append(meta, ignore_index=True)

        # Update progress
        current_idx += 1
        pbar.update(1)
    pbar.close()

    # Fix column ordering
    metadata = metadata[param_list]
    metadata_path = os.path.join(save_dir, 'metadata.csv')
    metadata.to_csv(metadata_path, index=None)
    print("Labels include: ", metadata.columns.values)

if __name__ == '__main__':
    main()
