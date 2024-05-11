#!/usr/bin/env python

"""Setup script."""

from setuptools import setup, find_packages

# Package name
name = 'vae_psf'

packages = find_packages()

# Scripts (in scripts/)
# scripts = ["scripts/deepify_interp"]
scripts = []

package_data = {}

setup(name=name,
      description=("vae_psf"),
      classifiers=["Topic :: Scientific :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      packages=packages,
      scripts=scripts)