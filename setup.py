from setuptools import find_packages, setup

setup(
    name='hidden',
    packages=find_packages(),
    version="0.1.0",
    desription='''Implementation of methods used for analysis of hidden markov
                  models, both on the system identification (parameter fitting)
                  and inferrence (filtering)''',
    author='slarge',
    license='MIT'
)
