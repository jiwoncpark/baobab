import numpy as np
import lenstronomy.Util.param_util as param_util

class DiagonalBNNPrior:
    def __init__(self, bnn_omega, components):
        self.components = components
        for comp in bnn_omega:
            if comp in self.components:
                # e.g. self.lens_mass = cfg.bnn_omega.lens_mass
                setattr(self, comp, bnn_omega[comp])

    def sample(self):
        kwargs = dict(
                      spemd=self._sample_spemd(),
                      ext_shear=self._sample_ext_shear(),
                      src_sersic=self._sample_src_sersic(),
                      )
        if 'lens_light' in self.components:
            kwargs['lens_sersic'] = self._sample_lens_sersic()
        if 'agn_light' in self.components:
            kwargs['agn_ps'] = self._sample_agn_ps()
        return kwargs

    def _sample_spemd(self):
        om = self.lens_mass # convenience renaming (from \Omega)
        x = np.random.normal(om.x.mu, om.x.sigma)
        y = np.random.normal(om.y.mu, om.y.sigma)
        gamma = np.random.lognormal(om.gamma.mu, om.gamma.sigma)
        theta_E = np.random.lognormal(om.theta_E.mu, om.theta_E.sigma)
        e1 = np.random.beta(om.e1.a, om.e1.b)
        e1 = e1*(om.e1.max - om.e1.min) + om.e1.min
        e2 = np.random.beta(om.e2.a, om.e2.b)
        e2 = e2*(om.e2.max - om.e2.min) + om.e2.min
        kwargs = {'center_x': x, 'center_y': y, 'gamma': gamma, 'theta_E': theta_E, 'e1': e1, 'e2': e2,}
        return kwargs

    def _sample_ext_shear(self):
        om = self.lens_mass # convenience renaming
        gamma_ext1 = np.random.beta(om.gamma_ext1.a, om.gamma_ext1.b)
        gamma_ext1 = gamma_ext1*(om.gamma_ext1.max - om.gamma_ext1.min) + om.gamma_ext1.min
        gamma_ext2 = np.random.beta(om.gamma_ext2.a, om.gamma_ext2.b)
        gamma_ext2 = gamma_ext2*(om.gamma_ext2.max - om.gamma_ext2.min) + om.gamma_ext2.min
        phi_ext, gamma_ext = param_util.ellipticity2phi_gamma(gamma_ext1, gamma_ext2)
        kwargs = {'psi_ext': phi_ext, 'gamma_ext': gamma_ext}
        return kwargs

    def _sample_src_sersic(self):
        om = self.src_light
        amp = np.random.lognormal(om.amp.mu, om.amp.sigma)
        x = np.random.normal(om.x.mu, om.x.sigma)
        y = np.random.normal(om.y.mu, om.y.sigma)
        n_sersic = np.random.lognormal(om.n_sersic.mu, om.n_sersic.sigma)
        r_eff = np.random.lognormal(om.R_sersic.mu, om.R_sersic.sigma)
        e1 = np.random.beta(om.e1.a, om.e1.b)
        e1 = e1*(om.e1.max - om.e1.min) + om.e1.min
        e2 = np.random.beta(om.e2.a, om.e2.b)
        e2 = e2*(om.e2.max - om.e2.min) + om.e2.min
        kwargs = {'amp': amp, 'center_x': x, 'center_y': y, 'n_sersic': n_sersic, 'R_sersic': r_eff, 'e1': e1, 'e2': e2}
        return kwargs

    def _sample_lens_sersic(self):
        om = self.lens_light
        amp = np.random.lognormal(om.amp.mu, om.amp.sigma)
        n_sersic = np.random.lognormal(om.n_sersic.mu, om.n_sersic.sigma)
        r_eff = np.random.lognormal(om.R_sersic.mu, om.R_sersic.sigma)
        e1 = np.random.beta(om.e1.a, om.e1.b)
        e1 = e1*(om.e1.max - om.e1.min) + om.e1.min
        e2 = np.random.beta(om.e2.a, om.e2.b)
        e2 = e2*(om.e2.max - om.e2.min) + om.e2.min
        kwargs = {'amp': amp, 'n_sersic': n_sersic, 'R_sersic': r_eff, 'e1': e1, 'e2': e2}
        return kwargs

    def _sample_agn_ps(self):
        om = self.agn_light
        amp = np.random.lognormal(om.amp.mu, om.amp.sigma)
        kwargs = {'amp': amp}
        return kwargs

        




