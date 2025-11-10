# DensityFlows

[![Build Status](https://github.com/gaetanfacchinetti/DensityFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gaetanfacchinetti/DensityFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)


A lightweight Julia package based on Flux.jl for scientists who want to emulate probability distributions efficiently using **normalizing flows**. Ready to flow?

---

</br>

<p align="center">
  <img src="docs/src/images/distrib.svg" width="430" height="450" />
</p>

## Installation

### Recommended installation

The code is not yet available on the global registry, refer to the section below for early installation.

### For developpers

To develop DensityFlows, you can clone this github repository

```bash
git clone https://github.com/gaetanfacchinetti/DensityFlows.jl.git
```
and install it in a julia shell
```julia
using Pkg; Pkg.develop(path = "<path>/DensityFlows.jl")
```

## Documentation

A public API as well as a manual can be found here: 
[documentation](https://gaetanfacchinetti.github.io/docs/DensityFlows.jl/).