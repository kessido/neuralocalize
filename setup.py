from distutils.core import setup

setup(
    name='Localize',
    version='0.1dev',
    packages=['python'], # TODO(loya) see if this name needs to be changed
    scripts=['python/src/localize'], # TODO(loya) validate and rename. Might also need .py
    long_description='''
    This package implements the model training and prediction described in Tavor el al,
    Task-free MRI predicts individual differences in brain activity during task performance:
    http://science.sciencemag.org/content/352/6282/216.
    ''',
)