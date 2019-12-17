import os
import shutil
import subprocess
import unittest

def test_to_hdf5():
    """Tests execution of `to_hdf5.py` script for all template config files
     
    """
    import baobab.configs as configs
    cfg_filepath = configs.tdlmc_diagonal_config.__file__
    cfg = configs.BaobabConfig.from_file(cfg_filepath)
    subprocess.check_output('generate {:s} --n_data 5'.format(cfg_filepath), shell=True)
    save_dir = cfg.out_dir
    n_failures = 0
    for channel_format in ['tf', 'theano']:
        try:
            subprocess.check_output('to_hdf5 {:s} --format {:s}'.format(save_dir, channel_format), shell=True)
        except:
            n_failures += 1
    # Delete resulting data
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    assert n_failures == 0 # FIXME: clumsy
            
if __name__ == '__main__':
    unittest.main()