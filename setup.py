from setuptools import setup, find_packages
#print(find_packages())
reqs = []
with open('requirements.txt') as f:
    reqs = f.read().splitlines()
#print(reqs)

setup(
      name='baobab',
      version='0.1',
      author='Ji Won Park',
      author_email='jiwon.christine.park@gmail.com',
      packages=find_packages(),
      license='LICENSE.md',
      description='Data generator for hierarchical inference with Bayesian neural networks',
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/jiwoncpark/baobab',
      install_requires=reqs,
      dependency_links=[
      'http://github.com/sibirrer/fastell4py/tarball/master#egg=fastell4py',],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['Development Status :: 4 - Beta',
      'License :: OSI Approved :: BSD License',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python'],
      keywords='physics'
      )