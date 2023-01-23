# Experiments for cell2sentence

This repository contains code to reproduce the findings of the `cell2sentence`
manuscript.  Preprint available
[here](https://www.biorxiv.org/content/10.1101/2022.09.18.508438).

## Environment Setup
This repository has been tested within an isolated conda environment
using `python` version `3.9.x` and `R` version 
`4.1.2 (2021-11-01) -- "Bird Hippie"` on macOS Monterey version `12.2`.

To configure your environment, install `R`, and set up a conda environment with

    conda create -n c2s python=3.9

Then activate with

    conda activate c2s

And finalize environment setup with:

    make setup

which will install all required dependencies for downstream analysis

## Reproducing Experiments

### PBMC Edit Distance Experiment




