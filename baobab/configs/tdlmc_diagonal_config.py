import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'tdlmc'
cfg.destination_dir = '.'
cfg.seed = 1113 # random seed
cfg.bnn_prior_class = 'DiagonalBNNPrior'
cfg.n_data = 200 # number of images to generate
cfg.train_vs_val = 'train'
cfg.components = ['lens_mass', 'external_shear', 'src_light', 'lens_light', 'agn_light']
cfg.checkpoint_interval = 2

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
                                          sigma=0.1,
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=0.1,
                                          ),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='normal',
                                              mu=2.0,
                                              sigma=0.1,
                                              ),
                                 theta_E = dict(
                                                dist='normal',
                                                mu=1.0,
                                                sigma=0.1,
                                                ),
                                 # Beta(a, b)
                                 q = dict(
                                           dist='normal',
                                           mu=0.7,
                                           sigma=0.15,
                                           lower=0.0,
                                           upper=0.3
                                           ),
                                 phi = dict(
                                           dist='uniform',
                                           lower=-0.5*np.pi,
                                           upper=0.5*np.pi,
                                           )
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='normal',
                                                         mu=0.008, # See overleaf doc
                                                         sigma=0.001,
                                                         ),
                                       psi_ext = dict(
                                                     dist='uniform',
                                                     lower=-0.5*np.pi,
                                                     upper=0.5*np.pi,
                                                     )
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  # Lognormal(mu, sigma^2)
                                  magnitude = dict(
                                             dist='normal',
                                             mu=18,
                                             sigma=1.0,
                                             ),
                                  n_sersic = dict(
                                                  dist='normal',
                                                  mu=3.0,
                                                  sigma=0.5,
                                                  ),
                                  R_sersic = dict(
                                                  dist='normal',
                                                  mu=0.7,
                                                  sigma=0.05,
                                                  
                                                  ),
                                  # Beta(a, b)
                                  q = dict(
                                           dist='normal',
                                           mu= 0.7,
                                           sigma=0.2,
                                           lower=0.1,
                                           upper=1.0
                                           ),
                                  phi = dict(
                                           dist='uniform',
                                           lower=-0.5*np.pi,
                                           upper=0.5*np.pi,
                                           )
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now
                                # Lognormal(mu, sigma^2)
                                magnitude = dict(
                                             dist='normal',
                                             mu=20.407,
                                             sigma=1,
                                             ),
                                n_sersic = dict(
                                                dist='lognormal',
                                                mu=0.7,
                                                sigma=0.4,
                                                ),
                                R_sersic = dict(
                                                dist='normal',
                                                mu=0.4,
                                                sigma=0.01,
                                                ),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                         dist='normal',
                                                mu=0,
                                                sigma=0.02,
                                                
                                                ),
                                center_y = dict(
                                         dist='normal',
                                                mu=0,
                                                sigma=0.02,
                                                
                                                ),
                                q = dict(
                                           dist='normal',
                                           mu=0.5,
                                           sigma=0.3,
                                           upper=1.0,
                                           lower=0.1
                                           ),
                                 phi = dict(
                                           dist='uniform',
                                           lower=-0.5*np.pi,
                                           upper=0.5*np.pi,
                                           )
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
                 )