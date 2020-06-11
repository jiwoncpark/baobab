import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'tdlmc'
cfg.seed = 1113 # random seed
cfg.bnn_prior_class = 'DiagonalCosmoBNNPrior'
cfg.n_data = 200 # number of images to generate
cfg.train_vs_val = 'train'
cfg.components = ['lens_mass', 'external_shear', 'src_light', 'lens_light', 'agn_light']

cfg.selection = dict(
                 magnification=dict(
                                    min=1.0
                                    ),
                 initial=["lambda x: x['lens_mass']['theta_E'] > 0.5",]
                 )

cfg.instrument = dict(
              pixel_scale=0.08, # scale (in arcseonds) of pixels
              ccd_gain=1.5, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              )

cfg.bandpass = dict(
                magnitude_zero_point=25.9463, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

cfg.observation = dict(
                  exposure_time=9600.0, # exposure time per image (in seconds)
                  #sky_brightness=20.1, # sky brightness (in magnitude per square arcseconds)
                  #num_exposures=10, # number of exposures that are combined
                  )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           #fwhm=0.1, # # full width at half maximum of the PSF (if not specific psf_model is specified)
           which_psf_maps=None, # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=1)

cfg.image = dict(
             num_pix=99, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

cfg.bnn_omega = dict(
                 # Inference hyperparameters defining the cosmology
                 cosmology = dict(
                                  H0=74.151, # Hubble constant at z = 0, in [km/sec/Mpc]
                                  Om0=0.27, # Omega matter: density of non-relativistic matter in units of the critical density at z=0.
                                  ),
                 # Hyperparameters of lens and source redshifts
                 redshift = dict(
                                model='independent_dist',
                                # Grid of redshift to sample from
                                z_lens=dict(
                                            dist='normal',
                                            mu=0.5,
                                            sigma=0.2,
                                            lower=0.0
                                            ),
                                z_src=dict(
                                           dist='normal',
                                           mu=2.0,
                                           sigma=0.4,
                                           lower=0.0
                                           ),
                                min_diff=1.0,
                                ),
                 # Hyperparameters of line-of-sight structure
                 LOS = dict(
                            kappa_ext = dict(
                                            dist='normal', # one of ['normal', 'beta']
                                            mu=0.0,
                                            sigma=0.025,
                                             ),
                            ),
                 # Hyperparameters and numerics for inferring the velocity dispersion for a given lens model
                 kinematics = dict(
                                   calculate_vel_disp=True,
                                   vel_disp_frac_err_sigma=0.05,
                                   anisotropy_model='analytic',
                                   kwargs_anisotropy=dict(
                                                          aniso_param=1.0
                                                          ),
                                   kwargs_aperture=dict(
                                                        aperture_type='slit',
                                                        center_ra=0.0,
                                                        center_dec=0.0,
                                                        width=1.0, # arcsec
                                                        length=1.0, # arcsec
                                                        angle=0.0,
                                                        ),
                                   kwargs_psf=dict(
                                                  psf_type='GAUSSIAN',
                                                  fwhm=0.6
                                                  ),
                                   kwargs_numerics=dict(
                                                       sampling_number=1000,
                                                       interpol_grid_num=1000,
                                                       log_integration=True,
                                                       max_integrate=100,
                                                       min_integrate=0.001
                                                       ),
                                   ),
                 time_delays = dict(
                                    calculate_time_delays=True,
                                    error_sigma=0.25,
                                    #frac_error_sigma=0.1,
                                    ),
                 lens_mass = dict(
                                 profile='PEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=1.e-7,
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-7,
                                          ),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='normal',
                                              mu=2.0,
                                              sigma=0.1,
                                              ),
                                 theta_E = dict(
                                                dist='normal',
                                                mu=1.1,
                                                sigma=0.05,
                                                ),
                                 # Beta(a, b)
                                 q = dict(
                                           dist='generalized_normal',
                                           mu=0.85,
                                           alpha=0.1,
                                           p=10.0,
                                           upper=1.0,
                                           lower=0.0,
                                           ),
                                 phi = dict(
                                           dist='generalized_normal',
                                           mu=0.0,
                                           alpha=np.pi*0.5,
                                           lower=-np.pi*0.5,
                                           p=10.0,
                                           upper=np.pi*0.5,
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                        dist='generalized_normal',
                                                        mu=0.025,
                                                        alpha=0.025,
                                                        lower=0.0,
                                                        p=10.0,
                                                        ),
                                       psi_ext = dict(
                                           dist='generalized_normal',
                                           mu=0.0,
                                           alpha=np.pi*0.5,
                                           lower=-np.pi*0.5,
                                           p=10.0,
                                           upper=np.pi*0.5,
                                           ),
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  # Lognormal(mu, sigma^2)
                                  magnitude = dict(
                                             dist='generalized_normal',
                                           mu=18.0,
                                           alpha=0.5,
                                           lower=0.0,
                                           p=10.0,
                                           ),
                                  n_sersic = dict(
                                                  dist='normal',
                                                  mu=3.0,
                                                  sigma=0.55,
                                                  ),
                                  R_sersic = dict(
                                                  dist='normal',
                                                  mu=0.8,
                                                  sigma=0.15,
                                                  lower=0.0,
                                                  ),
                                  # Beta(a, b)
                                  q = dict(
                                           dist='normal',
                                           mu= 0.85,
                                           sigma=0.1,
                                           upper=1.0,
                                           ),
                                  phi = dict(
                                           dist='generalized_normal',
                                           mu=0.0,
                                           alpha=np.pi*0.5,
                                           lower=-np.pi*0.5,
                                           p=10.0,
                                           upper=np.pi*0.5,
                                           ),
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now
                                # Lognormal(mu, sigma^2)
                                magnitude = dict(
                                             dist='normal',
                                             mu=21.0,
                                             sigma=0.7,
                                             ),
                                n_sersic = dict(
                                                dist='normal',
                                                mu=3.0,
                                                sigma=0.5,
                                                lower=0.0,
                                                ),
                                R_sersic = dict(
                                                dist='normal',
                                                mu=0.35,
                                                sigma=0.03,
                                                
                                                lower=0.0,
                                                ),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                                dist='generalized_normal',
                                                mu=0.0,
                                                alpha=0.07,
                                                p=10.0,
                                                ),
                                center_y = dict(
                                                dist='generalized_normal',
                                                mu=0.0,
                                                alpha=0.07,
                                                p=10.0,
                                                ),
                                q = dict(
                                         dist='normal',
                                         mu= 0.6,
                                         sigma=0.15,
                                         upper=1.0,
                                         lower=0.3,
                                         ),
                                phi = dict(
                                           dist='generalized_normal',
                                           mu=0.0,
                                           alpha=np.pi*0.5,
                                           lower=-np.pi*0.5,
                                           p=10.0,
                                           upper=np.pi*0.5,
                                           ),
                                ),

                 agn_light = dict(
                                 profile='LENSED_POSITION', # contains one of 'LENSED_POSITION' or 'SOURCE_POSITION'
                                 # Centered at host
                                 # Pre-magnification, image-plane amplitudes if 'LENSED_POSITION'
                                 # Lognormal(mu, sigma^2)
                                 magnitude = dict(
                                             dist='normal',
                                             mu=21.25,
                                             sigma=0.7,
                                             lower=0.0,
                                             ),
                                 ),
                 )