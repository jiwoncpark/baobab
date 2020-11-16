import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'tdlmc'
cfg.destination_dir = '.'
cfg.seed = 1113 # random seed
cfg.bnn_prior_class = 'CovBNNPrior'
cfg.n_data = 200 # number of images to generate
cfg.train_vs_val = 'train'
cfg.components = ['lens_mass', 'external_shear', 'src_light', 'lens_light', 'agn_light']
cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 initial=["lambda x: x['lens_mass']['theta_E'] > 0.5",]
                 )

cfg.survey_info = dict(
                       survey_name="HST",
                       bandpass_list=["WFC3_F160W"]
                       )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           which_psf_maps=[101], # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=1)

cfg.image = dict(
             num_pix=100, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

cfg.bnn_omega = dict(
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
                                              dist='lognormal',
                                              mu=0.7,
                                              sigma=0.02,
                                              ),
                                 theta_E = dict(
                                                dist='lognormal',
                                                mu=0.0,
                                                sigma=0.1,
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
                                           upper=0.9,),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu=-2.73, # See overleaf doc
                                                         sigma=1.05
                                                         ),
                                       psi_ext = dict(
                                                     dist='generalized_normal',
                                                     mu=0.0,
                                                     alpha=0.5*np.pi,
                                                     p=10.0,
                                                     lower=-0.5*np.pi,
                                                     upper=0.5*np.pi
                                                     ),
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
                                             ),
                                  n_sersic = dict(
                                                  dist='lognormal',
                                                  mu=1.25,
                                                  sigma=0.13,
                                                  ),
                                  R_sersic = dict(
                                                  dist='lognormal',
                                                  mu=-0.35,
                                                  sigma=0.3,
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

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now
                                # Lognormal(mu, sigma^2)
                                magnitude = dict(
                                             dist='normal',
                                             mu=22,
                                             sigma=1,
                                             lower=0.0,
                                             ),
                                n_sersic = dict(
                                                dist='lognormal',
                                                mu=0.7,
                                                sigma=0.4,
                                                ),
                                R_sersic = dict(
                                                dist='lognormal',
                                                mu=-0.7,
                                                sigma=0.4,
                                                ),
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
                                             ),
                                 ),
                 cov_info = dict(
                                # List of 2-tuples specifying which params are correlated 
                                cov_params_list=[                     
                                ('lens_mass', 'gamma'),
                                ('lens_light', 'n_sersic'),
                                ],
                                cov_omega = dict(
                                                # Whether each param is log-parameterized
                                                is_log=[
                                                True,
                                                True,
                                                ],
                                                # The mean vector
                                                mu=[0.7, 1.25],
                                                # The covariance matrix (must be PSD and symmetric numpy array)
                                                cov_mat=[[0.1, 0.03], [0.03, 0.4]],
                                                lower=None,
                                                upper=None,
                                                 ),
                                )
                 )

