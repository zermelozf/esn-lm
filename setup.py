#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='esn-lm',
      version='0.1',
      description='Language modeling with echo state networks',
      author='Arnaud Rachez',
      author_email='arnaud.rachez@gmail.com',
      url='http://soft.ics.keio.ac.jp/~arnaud',
      packages=find_packages(exclude=['ez_setup']),
      install_requires=['setuptools'],
      test_suite='nose.collector',
      test_requires=['Nose'],
      )

