import os, sys
import warnings
from importlib import import_module
from addict import Dict

class BaobabConfig:
    """Nested dictionary representing the configuration for Baobab data generation
    
    """
    def __init__(self, user_cfg):
        """
        Parameters
        ----------
        user_cfg : dict or Dict
            user-defined configuration
        
        """
        self.__dict__ = Dict(user_cfg)
        self.interpret_kinematics_cfg()

    def interpret_kinematics_cfg(self):
        """Validate the kinematics config

        """
        kinematics_cfg = self.bnn_omega.kinematics_cfg
        if kinematics_cfg.anisotropy_model == 'analytic':
            warnings.warn("Since velocity dispersion computation is analytic, any entry other than `sampling_number` in `kinematics.numerics_kwargs` will be ignored.")

    @classmethod
    def from_file(cls, user_cfg_path):
        """Alternative constructor that accepts the path to the user-defined configuration python file
        
        Parameters
        ----------
        user_cfg_path : str or os.path object
            path to the user-defined configuration python file
        
        """
        dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
        module_name, ext = os.path.splitext(filename)
        sys.path.insert(0, dirname)
        #user_cfg_file = map(__import__, module_name)
        #user_cfg = getattr(user_cfg_file, 'cfg')
        user_cfg_script = import_module(module_name)
        user_cfg = getattr(user_cfg_script, 'cfg')
        return cls(user_cfg)