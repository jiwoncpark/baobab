import astropy.io.fits as pyfits
from pkg_resources import resource_filename
from lenstronomy.Util import kernel_util
from lenstronomy.Data.psf import PSF
__all__ = ['instantiate_PSF_models', 'get_PSF_model']

def instantiate_PSF_models(psf_config, pixel_scale):
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
    if psf_config['type'] == 'PIXEL':
        psf_models = []
        if psf_config['which_psf_maps'] is None:
            # Instantiate PSF with all available PSF maps
            #FIXME: equate psf_id with psf_i since seed number is meaningless
            psf_id_list = [101, 150]
        else:
            psf_id_list = [psf_config['which_psf_maps']]

        for psf_i, psf_id in enumerate(psf_id_list):
            psf_path = resource_filename('baobab.in_data', 'psf_maps/psf_{:d}.fits'.format(psf_id))
            psf_map = pyfits.getdata(psf_path)
            kernel_cut = kernel_util.cut_psf(psf_map, psf_config['kernel_size'])
            kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': pixel_scale, 'kernel_point_source': kernel_cut}
            psf_models.append(PSF(**kwargs_psf))
        return psf_models
    elif psf_config['type'] == 'GAUSSIAN':
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_config['PSF_FWHM'], 'pixel_size': pixel_scale}
        psf_model = PSF(**kwargs_psf)
        return [psf_model]
    else:
        psf_model = PSF(psf_type='NONE')
        return [psf_model]

def get_PSF_model(psf_models, n_psf, current_idx):
    """Get a single PSF model from the model(s) previously instantiated
    
    Parameters
    ----------
    psf_model : list
        list of PSF model(s)
    n_psf : int
        number of PSF model(s)
    current_idx : int

    Returns
    -------
    lenstronomy.PSF instance
        a single PSF model

    """
    return psf_models[current_idx%n_psf]