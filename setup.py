from setuptools import setup, find_packages

requirements = [
    'numpy',
    'scikit-learn'
]

VERSION = '0.0.1'

setup(
    name='grd',
    version=VERSION,
    url='https://github.com/vishwakftw/Graded-Relations-From-Data',
    description="Implementation of the framework in the paper: Waegeman, W., Pahikkala, T., Airola, A., Salakoski, T., Stock, M., & De Baets, B. (2012). A kernel-based framework for learning graded relations from data. IEEE Transactions on Fuzzy Systems, 20(6), 1090-1101.",
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements
)