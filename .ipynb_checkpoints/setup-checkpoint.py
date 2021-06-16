from setuptools import find_packages, setup

install_requires = [
    'pandas',
    'numpy',
    'inflection',
    'functools',
    'pickle-mixin',
    'datetime',
    'math',
    'scikit-learn'
]

# tests_require = ['pytest>=4.0.2']

setup(name='rossmann',
      version='0.0.1',
      description='App de previsão de vendas',
      author='Bruno Vinicius Nonato',
      author_email='brunovinicius154@gmail.com',
      install_requires=install_requires,
      packages=['rossmann'])