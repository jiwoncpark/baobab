import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'gamma'
cfg.seed = 1113 # random seed
cfg.bnn_prior_class = 'EmpiricalBNNPrior'
cfg.n_data = 200 # number of images to generate
cfg.train_vs_val = 'train'
cfg.components = ['lens_mass', 'external_shear', 'src_light']

cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 initial=["lambda x: x['lens_mass']['theta_E'] > 1.0",
                 "lambda x: x['lens_mass']['theta_E'] < 1.6"]
                 )

cfg.instrument = dict(
              pixel_scale=0.08, # scale (in arcseonds) of pixels
              ccd_gain=4.5, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              )

cfg.bandpass = dict(
                magnitude_zero_point=25.9463, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

cfg.observation = dict(
                  exposure_time=100.0, # exposure time per image (in seconds)
                  )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
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
                                  H0=70.0, # Hubble constant at z = 0, in [km/sec/Mpc]
                                  Om0=0.3, # Omega matter: density of non-relativistic matter in units of the critical density at z=0.
                                  ),
                 redshift = dict(
                                model='differential_comoving_volume',
                                # Grid of redshift to sample from
                                grid = dict(
                                            start=0.1, # min redshift
                                            stop=4.0, # max redshift
                                            step=0.05, # resolution of the z_grid
                                            ),
                                min_diff=0.0,
                                ),

                 kinematics = dict(
                                   # Grid of velocity dispersion to sample from
                                   vel_disp = dict(
                                                  model = 'vel_disp_function_CPV2007', # one of ['vel_disp_function_CPV2007',] -- see docstring for details 
                                                  grid = dict(
                                                             start=200.0, # km/s
                                                             stop=400.0, # km/s
                                                             step=5.0, # km/s
                                                             ),
                                                  )
                                   ),
                 lens_mass = dict(
                                 profile='PEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=0.07,
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=0.07,
                                          ),
                                 gamma = dict(
                                              model='FundamentalMassHyperplane',
                                              model_kwargs = dict(
                                                                  fit_data='SLACS',
                                                                  ),
                                              ),
                                 theta_E = dict(
                                                model='approximate_theta_E_for_SIS',
                                                ),
                                 # Beta(a, b)
                                 q = dict(
                                           model='AxisRatioRayleigh',
                                           model_kwargs = dict(
                                                             fit_data='SDSS'
                                                             ),
                                           ),
                                 phi = dict(
                                           dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu=-2.73, # See overleaf doc
                                                         sigma=0.2,
                                                         ),
                                       psi_ext = dict(
                                                     dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5
                                                     ),
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  magnitude = dict(
                                                   model='FaberJackson',
                                                   model_kwargs = dict(
                                                                     fit_data='ETGs',
                                                                     ),
                                                   ),
                                  R_sersic = dict(
                                                  model='FundamentalPlane',
                                                  model_kwargs = dict(
                                                                    fit_data='SDSS',),
                                                  ),
                                  n_sersic = dict(
                                                  dist='lognormal',
                                                  mu=1.25,
                                                  sigma=0.13,
                                                  ),
                                  # axis ratio
                                  q = dict(
                                           model='AxisRatioRayleigh',
                                           model_kwargs = dict(
                                                             fit_data='SDSS'
                                                             ),
                                           ),
                                  # ellipticity angle
                                  phi = dict(
                                             dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5
                                             ),
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now 
                                magnitude = dict(
                                                 model='redshift_binned_luminosity_function',
                                                 ),
                                n_sersic = dict(
                                                dist='lognormal',
                                                mu=0.8,
                                                sigma=0.3,
                                                ),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                         dist='uniform',
                                         lower=-0.1,
                                         upper=0.1,
                                         ),
                                center_y = dict(
                                               dist='uniform',
                                         lower=-0.1,
                                         upper=0.1,    
                                                ),
                                R_sersic = dict(
                                                model='size_from_luminosity_and_redshift_relation',
                                                ),
                                q = dict(
                                         dist='one_minus_rayleigh',
                                         scale=0.3,
                                         lower=0.2
                                         ),
                                phi = dict(
                                           dist='uniform',
                                           lower=-np.pi*0.5,
                                           upper=np.pi*0.5
                                           ),
                                ),

                 agn_light = dict(
                                 profile='LENSED_POSITION', # contains one of 'LENSED_POSITION' or 'SOURCE_POSITION'
                                 # Centered at host
                                 # Pre-magnification, image-plane amplitudes if 'LENSED_POSITION'
                                 magnitude = dict(
                                                 model='AGNLuminosityFunction',
                                                 model_kwargs = dict(
                                                                     M_grid=np.arange(-30.0, -19.0, 0.2).tolist(),
                                                                     fit_data='combined',
                                                                     ),
                                                 ),
                                 ),
                 )