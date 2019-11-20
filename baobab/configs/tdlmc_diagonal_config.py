import os, sys
import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'tdlmc'
cfg.seed = 1113 # random seed
cfg.bnn_prior_class = 'DiagonalBNNPrior'
cfg.n_data = 200 # number of images to generate
cfg.train_vs_val = 'train'
cfg.out_dir = os.path.join('{:s}_{:s}_{:s}_seed{:d}'.format(cfg.name,
                                                        cfg.train_vs_val,
                                                        cfg.bnn_prior_class,
                                                        cfg.seed))
cfg.components = ['lens_mass', 'external_shear', 'src_light', 'lens_light', 'agn_light']

cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 theta_E=dict(
                              min=0.5,
                              ),
                 )

cfg.instrument = dict(
              pixel_scale=0.08, # scale (in arcseonds) of pixels
              ccd_gain=4.5, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              read_noise=10, # std of noise generated by read-out (in units of electrons)
              )

cfg.bandpass = dict(
                magnitude_zero_point=25.9463, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

cfg.observation = dict(
                  exposure_time=100.0, # exposure time per image (in seconds)
                  sky_brightness=20.1, # sky brightness (in magnitude per square arcseconds)
                  num_exposures=10, # number of exposures that are combined
                  background_noise=0.2, # overrides exposure_time, sky_brightness, read_noise, num_exposures
                  )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           fwhm=0.1, # # full width at half maximum of the PSF (if not specific psf_model is specified)
           which_psf_maps=None, # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=1)

cfg.image = dict(
             num_pix=100, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

cfg.bnn_omega = dict(
                 # Inference hyperparameters defining the cosmology
                 cosmology = dict(
                                  H0=74.151, # Hubble constant at z = 0, in [km/sec/Mpc]
                                  Om0=0.27, # Omega matter: density of non-relativistic matter in units of the critical density at z=0.
                                  Ode0=0.73, # Omega dark energy: density of dark energy in units of the critical density at z=0.
                                  w0=-1.0
                                  ), # Dark energy equation of state at all redshifts. This is pressure/density for dark energy in units where c=1. A cosmological constant has w0=-1.0.
                 # Hyperparameters of lens and source redshifts
                 redshift = dict(
                                model='differential_comoving_volume',
                                # Grid of redshift to sample from
                                grid = dict(
                                            start=0.5, # min redshift
                                            stop=4.0, # max redshift
                                            step=0.1, # resolution of the z_grid
                                            ),
                                ),
                 # Hyperparameters of line-of-sight structure
                 LOS = dict(
                            kappa_ext = dict(
                                            dist='normal', # one of ['normal', 'beta']
                                            mu=0.0,
                                            sigma=0.025,
                                            log=False
                                             ),
                            ),
                 # Hyperparameters and numerics for inferring the velocity dispersion for a given lens model
                 kinematics = dict(
                                   # Grid of velocity dispersion to sample from
                                   vel_disp_frac_err_sigma=0.05,
                                   aniso_param=1.0, # r_ani = aniso_param*r_eff,
                                   aperture_size_x=1.0, # arcsec
                                   aperture_size_y=1.0, # arcsec
                                   psf_fwhm_eff=0.6, # arcsec
                                   num_evaluate=1000,
                                   ),
                 time_delays = dict(
                                    error_sigma=0.25,
                                    #frac_error_sigma=0.1,
                                    ),
                 lens_mass = dict(
                                 profile='SPEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=1.e-6,
                                          log=False),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-6,
                                          log=False),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='normal',
                                              mu=1.935,
                                              sigma=0.001,
                                              log=False),
                                 theta_E = dict(
                                                dist='normal',
                                                mu=1.082,
                                                sigma=0.001,
                                                log=False),
                                 # Beta(a, b)
                                 q = dict(
                                           dist='normal',
                                           mu=0.869,
                                           sigma=0.001,
                                           log=False,
                                           ),
                                 phi = dict(
                                           dist='normal',
                                           mu= 0.708,
                                           sigma=0.001,
                                           log=False,
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='normal',
                                                         mu=0.008, # See overleaf doc
                                                         sigma=0.001,
                                                         log=False,),
                                       psi_ext = dict(
                                                     dist='normal',
                                                     mu=0.7853,
                                                     sigma=0.001,
                                                     lower=0,
                                                     upper=np.pi,
                                                     log=False
                                                     )
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  # Lognormal(mu, sigma^2)
                                  magnitude = dict(
                                             dist='normal',
                                             mu=17.325,
                                             sigma=0.001,
                                             log=False),
                                  n_sersic = dict(
                                                  dist='normal',
                                                  mu=2.683,
                                                  sigma=0.001,
                                                  log=False),
                                  R_sersic = dict(
                                                  dist='normal',
                                                  mu=0.949,
                                                  sigma=0.001,
                                                  log=False
                                                  ),
                                  # Beta(a, b)
                                  q = dict(
                                           dist='normal',
                                           mu= 0.746,
                                           sigma=0.001,
                                           log=False,
                                           ),
                                  phi = dict(
                                            dist='normal',
                                           mu= 0.658,
                                           sigma=0.001,
                                           log=False,
                                           ),
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now
                                # Lognormal(mu, sigma^2)
                                magnitude = dict(
                                             dist='normal',
                                             mu=20.407,
                                             sigma=0.001,
                                             log=False),
                                n_sersic = dict(
                                                dist='normal',
                                                mu=0.7,
                                                sigma=0.4,
                                                log=True),
                                R_sersic = dict(
                                                dist='normal',
                                                mu=0.4,
                                                sigma=0.01,
                                                log=False
                                                ),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                         dist='normal',
                                                mu=0.035,
                                                sigma=0.001,
                                                log=False
                                                ),
                                center_y = dict(
                                         dist='normal',
                                                mu=-0.025,
                                                sigma=0.001,
                                                log=False
                                                ),
                                q = dict(
                                           dist='normal',
                                           mu=0.869,
                                           sigma=0.001,
                                           log=False,
                                           ),
                                 phi = dict(
                                           dist='normal',
                                           mu= 0.708,
                                           sigma=0.001,
                                           log=False,
                                           ),
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
                                             log=False
                                             ),
                                 ),
                 )