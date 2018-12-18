from setuptools import setup, find_packages

setup(name='bilangsim',
      python_requires='>=3.6',
      version='0.1',
      description='Agent-based simulator of bilingual societies',
      url='https://github.com/pgervila/bilingual_society_ABM/',
      author='Paolo Gervasoni Vila',
      author_email='pgervila@gmail.com',
      packages=find_packages(),
      install_requires=['mesa==0.7.8.1', 'numpy', 'scipy', 'numba', 'pandas', 'scikit-learn',
                        'matplotlib', 'networkx', 'pyprind', 'deepdish', 'dill'],
      include_package_data=True,
      package_data={'bilangsim': ['data/cdfs/*.h5', 'data/init_conds/*.h5']}
      )

__author__ = 'Paolo Gervasoni Vila'