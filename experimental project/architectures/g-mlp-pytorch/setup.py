from setuptools import setup, find_packages

setup(
  name = 'g-mlp-pytorch',
  packages = find_packages(),
  version = '0.1.5',
  license='MIT',
  description = 'gMLP - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/g-mlp-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'multi-layered-preceptrons'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
