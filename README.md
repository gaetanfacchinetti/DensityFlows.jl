# DensityFlows

[![Build Status](https://github.com/gaetanfacchinetti/DensityFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gaetanfacchinetti/DensityFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)


This module implements some simple **Normalizing Flows** based on Flux.jl. Any comment is welcome.

## Installation

### Recommended installation

The code is not yet available on the global registry, refer to the section below for early installation.

### For developpers

To develop CosmoTools, you can clone this github repository

```bash
git clone https://github.com/gaetanfacchinetti/DensityFlows.jl.git
```
and install it in a julia shell
```julia
using Pkg; Pkg.develop(path = "<path>/DensityFlows.jl")
```


## Quick start guide

A simple flow made of 3 blocks of 2 RNVP-coupling layers can be defined as

```julia
using DensityFlows
x = rand(2, 5000);
data = DataArrays(x)
flow = AffineCouplingFlow(3, data.metadata, n_sublayers_s=2, n_sublayers_t=2)
```


## Documentation

A public API as well as a preliminary manual can be found here: 
[documentation](https://gaetanfacchinetti.github.io/docs/DensityFlows.jl/).