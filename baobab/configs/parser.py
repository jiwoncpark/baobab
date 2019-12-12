import os, sys
from datetime import datetime
import warnings
from importlib import import_module
from addict import Dict
import json

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
        if not hasattr(self, 'out_dir'):
            # Default out_dir path if not specified
            self.out_dir = os.path.join(os.getcwd(), '{:s}_{:s}_prior={:s}_seed={:d}'.format(self.name, self.train_vs_val, self.bnn_prior_class, self.seed))
        self.out_dir = os.path.abspath(self.out_dir)
        self.checkpoint_interval = max(100, self.n_data // 100)
        self.interpret_kinematics_cfg()
        self.log_filename = datetime.now().strftime("log_%m-%d-%Y_%H:%M_baobab.json")
        self.log_path = os.path.join(self.out_dir, self.log_filename)

    def export_log(self):
        """Export the baobab log to the current working directory

        """
        with open(self.log_path, 'w') as f:
            json.dump(self.__dict__, f)
            print("Exporting baobab log to {:s}".format(self.log_path))

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