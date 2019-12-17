__all__ = ['Selection']

class Selection:
    """Selections applied to the sampled set of parameters

    """
    def __init__(self, selection_cfg, components):
        """
        Parameters
        ----------
        selection_cfg : dict
            copy of `cfg.selection`
        components: list
            list of components to render (copy of `cfg.components`)

        """
        self.components = components
        self.init_selections = [eval(s) for s in selection_cfg['initial']]
        self.init_selections += self.get_ellipticity_selections()

    def get_ellipticity_selections(self):
        """Get default selections for a self-consistent ellipticity definition

        Returns
        -------
        list
            lambda functions with the sample dictionary as the argument, each of which returns True if the selection passes

        """
        ellip_selections = []
        ellip_selections += [lambda x: ((x['lens_mass']['e1']**2.0 + x['lens_mass']['e2']**2.0)**0.5 < 1.0)]
        ellip_selections += [lambda x: ((x['src_light']['e1']**2.0 + x['src_light']['e2']**2.0)**0.5 < 1.0)]
        if 'lens_light' in self.components:
            ellip_selections += [lambda x: ((x['lens_light']['e1']**2.0 + x['lens_light']['e2']**2.0)**0.5 < 1.0)]
        return ellip_selections

    def reject_initial(self, sample):
        """Determine whether to reject the sample

        Parameters
        ----------
        sample : dict
            sampled parameters

        Returns
        -------
        bool
            whether to reject this sample

        """
        passed = [l(sample) for l in self.init_selections]
        if not all(passed):
            return True