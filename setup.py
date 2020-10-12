from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='ASTARS',
      version='0.0.1',
      description='derivative free optimizer exploiting the active subspace',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='dimension reduction mathematics active subspaces uncertainty quantification optmization',
      url='https://github.com/variscarey/ASTARS',
      author='Varis Carey and Jordan Hall',
      author_email='variscarey@googlemail.com',
      license='MIT',
      packages=['astars', 'astars.utils'],
      install_requires=[
          'numpy',
          'scipy >= 0.15.0',
          'matplotlib',
          'pandas'
      ],
      #test_suite='nose.collector',
      #tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
