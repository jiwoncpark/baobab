import astropy.io.fits as pyfits
import numpy as np
from pkg_resources import resource_filename
from lenstronomy.Util import kernel_util
from lenstronomy.Data.psf import PSF
__all__ = ['get_PSF_model']

def get_PSF_model(psf_type, pixel_scale, seeing, kernel_size, which_psf_maps):
    """Instantiate PSF kwargs by reading in template PSF maps

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
    if psf_type == 'PIXEL':
        psf_id_list = which_psf_maps
        random_psf_id = psf_id_list[np.random.randint(len(psf_id_list))]
        psf_path = resource_filename('baobab.in_data', 'psf_maps/psf_{:d}.fits'.format(random_psf_id))
        psf_map = pyfits.getdata(psf_path)
        kernel_cut = kernel_util.cut_psf(psf_map, kernel_size)
        psf_kwargs = {'psf_type': 'PIXEL', 'pixel_size': pixel_scale, 'kernel_point_source': kernel_cut}
        return PSF(**psf_kwargs)
    elif psf_type == 'GAUSSIAN':
        psf_kwargs = {'psf_type': 'GAUSSIAN', 'fwhm': seeing, 'pixel_size': pixel_scale}
        return PSF(**psf_kwargs)
    else:
        return PSF(psf_type='NONE')