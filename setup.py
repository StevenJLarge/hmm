from setuptools import find_packages, setup

# README contents
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hidden-py',
    packages=find_packages(),
    version="1.0.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
    desription='''
        Implementation of methods used for analysis of hidden markov
        models, both on the system identification (parameter fitting)
        and inferrence (filtering)
    ''',
    author='Steven Large',
    author_email='stevelarge7@gmail.com',
    license='MIT',
    install_requires=[
        'pytest',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn >= 0.11.1',
        'numba'
    ],
    project_urls={
        'Source': 'https://github.com/StevenJLarge/hmm'
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
)
