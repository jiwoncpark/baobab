import os
import shutil
import subprocess
import unittest

class TestGenerate(unittest.TestCase):
    """Test the `generate.py` script

    """

    def test_generate(self):
        """Tests execution of `generate.py` script for all template config files
         
        """
        import baobab.configs as configs
        cfg_root = os.path.abspath(os.path.dirname(configs.__file__))
        n_failures = 0
        for cfg_filename in os.listdir(cfg_root):
            if cfg_filename.endswith('_config.py'):
                cfg_filepath = os.path.join(cfg_root, cfg_filename)
                cfg = configs.BaobabConfig.from_file(cfg_filepath)
                save_dir = cfg.out_dir
                try:
                    subprocess.check_output('generate {:s} --n_data 2'.format(cfg_filepath), shell=True)
                except:
                    n_failures += 1
                # Delete resulting data
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)

        assert n_failures == 0 # FIXME: clumsy
            
if __name__ == '__main__':
    unittest.main()

        

    
