import os, sys

name = 'gamma'
seed = 1113 # random seed
bnn_prior_class = 'DiagonalBNNPrior'
n_data = 200 # number of images to generate
train_vs_val = 'train'
out_dir = os.path.join('out_data', '{:s}_{:s}_{:s}_seed{:d}'.format(name,
                                                                    train_vs_val,
                                                                    bnn_prior_class,
                                                                    seed))
components = ['lens_mass', 'external_shear', 'src_light',] #'lens_light', 'agn_light']

image = dict(
             sigma_bkg=0.05, 
             exposure_time=100.0, # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit),
             numPix=100, # cutout pixel size
             deltaPix=0.08, # arcsec/pix
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

psf = dict(
           type='PIXEL', # one of ['gaussian', 'PIXEL', 'NONE']
           kernel_dir=os.path.join('in_data', 'psf_maps'), # only valid when profile='PIXEL'
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           fwhm=0.1, # only valid when profile='gaussian'
           )

bnn_omega = dict(
                 lens_mass = dict(
                                 profile='SPEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=1.e-7),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-7),
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
                                                         dist='beta',
                                                         a=4.0,
                                                         b=4.0,
                                                         lower=-0.2,
                                                         upper=0.2),
                                       psi_ext = dict(
                                                     dist='beta',
                                                     a=4.0,
                                                     b=4.0,
                                                     lower=-0.2,
                                                     upper=0.2)
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  # Lognormal(mu, sigma^2)
                                  amp = dict(
                                             dist='normal',
                                             mu=3.9,
                                             sigma=0.7,
                                             log=True),
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
                                amp = dict(
                                           dist='normal',
                                           mu=5.0,
                                           sigma=0.3,
                                           log=True),
                                n_sersic = dict(
                                                dist='normal',
                                                mu=1.1,
                                                sigma=0.2,
                                                log=True),
                                R_sersic = dict(
                                                dist='normal',
                                                mu=-0.7,
                                                sigma=0.4,
                                                log=True),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                         dist='normal',
                                         mu=0.0,
                                         sigma=0.01),
                                center_y = dict(
                                         dist='normal',
                                         mu=0.0,
                                         sigma=0.01),
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
                                 amp = dict(
                                            dist='normal',
                                            mu=3.5, 
                                            sigma=0.3,
                                            log=True),
                                 ),

                 cosmo = dict(
                             # Normal(mu, sigma^2)
                             z_lens = dict(
                                           dist='normal',
                                           mu=1.5,
                                           sigma=0.2,
                                           lower=0.1,
                                           upper=2.5),
                             z_src = dict(
                                          dist='normal',
                                          mu=1.5,
                                          sigma=0.2,
                                          lower=-1,
                                          upper=99),
                             # Uniform
                             H0 = dict(
                                       lower=50.0,
                                       upper=90.0),
                             # Uniform (scaled by r_eff)
                             r_ani = dict(
                                          lower=0.5,
                                          upper=5.0),
                             ),
                 )