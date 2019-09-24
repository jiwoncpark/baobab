import os, sys
import numpy as np

name = 'tdlmc'
seed = 1113 # random seed
bnn_prior_class = 'DiagonalBNNPrior'
n_data = 200 # number of images to generate
train_vs_val = 'train'
out_dir = os.path.join('out_data', '{:s}_{:s}_{:s}_seed{:d}'.format(name,
                                                                    train_vs_val,
                                                                    bnn_prior_class,
                                                                    seed))
components = ['lens_mass', 'external_shear', 'src_light', 'lens_light', 'agn_light']

selection = dict(
                 magnification=dict(
                                    min=2.0))

instrument = dict(
              pixel_scale=0.08, # scale (in arcseonds) of pixels
              ccd_gain=4.5, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              read_noise=10, # std of noise generated by read-out (in units of electrons)
              )

bandpass = dict(
                magnitude_zero_point=25.9463, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

observation = dict(
                  exposure_time=100.0, # exposure time per image (in seconds)
                  sky_brightness=20.1, # sky brightness (in magnitude per square arcseconds)
                  num_exposures=10, # number of exposures that are combined
                  background_noise=0.25, # overrides exposure_time, sky_brightness, read_noise, num_exposures
                  )

psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           fwhm=0.1, # # full width at half maximum of the PSF (if not specific psf_model is specified)
           which_psf_maps=None, # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

numerics = dict(
                supersampling_factor=1)

image = dict(
             num_pix=100, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

bnn_omega = dict(
                 lens_mass = dict(
                                 profile='SPEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=1.e-7,
                                          log=False),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-7,
                                          log=False),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='normal',
                                              mu=0.7,
                                              sigma=0.02,
                                              log=True),
                                 theta_E = dict(
                                                dist='normal',
                                                mu=0.0,
                                                sigma=0.1,
                                                log=True),
                                 # Beta(a, b)
                                 e1 = dict(
                                           dist='beta',
                                           a=4.0,
                                           b=4.0,
                                           lower=-0.9,
                                           upper=0.9),
                                 e2 = dict(
                                           dist='beta',
                                           a=4.0,
                                           b=4.0,
                                           lower=-0.9,
                                           upper=0.9,),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='normal',
                                                         mu=-2.73, # See overleaf doc
                                                         sigma=1.05,
                                                         log=True,),
                                       psi_ext = dict(
                                                     dist='generalized_normal',
                                                     mu=np.pi,
                                                     alpha=np.pi,
                                                     p=10.0,
                                                     lower=0.0,
                                                     upper=2.0*np.pi)
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  # Lognormal(mu, sigma^2)
                                  magnitude = dict(
                                             dist='normal',
                                             mu=15,
                                             sigma=1,
                                             lower=0.0,
                                             log=False),
                                  n_sersic = dict(
                                                  dist='normal',
                                                  mu=1.25,
                                                  sigma=0.13,
                                                  log=True),
                                  R_sersic = dict(
                                                  dist='normal',
                                                  mu=-0.35,
                                                  sigma=0.3,
                                                  log=True),
                                  # Beta(a, b)
                                  e1 = dict(
                                            dist='beta',
                                            a=4.0,
                                            b=4.0,
                                            lower=-0.9,
                                            upper=0.9),
                                  e2 = dict(
                                            dist='beta',
                                            a=4.0,
                                            b=4.0,
                                            lower=-0.9,
                                            upper=0.9),
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now
                                # Lognormal(mu, sigma^2)
                                magnitude = dict(
                                             dist='normal',
                                             mu=22,
                                             sigma=1,
                                             lower=0.0,
                                             log=False),
                                n_sersic = dict(
                                                dist='normal',
                                                mu=0.7,
                                                sigma=0.4,
                                                log=True),
                                R_sersic = dict(
                                                dist='normal',
                                                mu=-0.7,
                                                sigma=0.4,
                                                log=True),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                         dist='generalized_normal',
                                         mu=0.0,
                                         alpha=0.03,
                                         p=10.0,
                                         ),
                                center_y = dict(
                                         dist='generalized_normal',
                                         mu=0.0,
                                         alpha=0.03,
                                         p=10.0,
                                         ),
                                # Beta(a, b)
                                e1 = dict(
                                          dist='beta',
                                          a=4.0,
                                          b=4.0,
                                          lower=-0.9,
                                          upper=0.9),
                                e2 = dict(
                                          dist='beta',
                                          a=4.0,
                                          b=4.0,
                                          lower=-0.9,
                                          upper=0.9),
                                ),

                 agn_light = dict(
                                 profile='LENSED_POSITION', # contains one of 'LENSED_POSITION' or 'SOURCE_POSITION'
                                 # Centered at host
                                 # Pre-magnification, image-plane amplitudes if 'LENSED_POSITION'
                                 # Lognormal(mu, sigma^2)
                                 magnitude = dict(
                                             dist='normal',
                                             mu=21,
                                             sigma=1,
                                             lower=0.0,
                                             log=False),
                                 ),
                 )