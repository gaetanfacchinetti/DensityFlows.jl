# DensityFlows


[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://gaetanfacchinetti.github.io/docs/DensityFlows.jl/)  [![Build Status](https://github.com/gaetanfacchinetti/DensityFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gaetanfacchinetti/DensityFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)



A lightweight Julia package based on [Flux.jl](https://github.com/FluxML/Flux.jl) for scientists who want to emulate probability distributions efficiently using **normalizing flows**. Designed for users with no prior experience, yet flexible enough to let you build and customize your own layers. Ready to flow?


## 🚀 Features

- RealNVP affine coupling layers
- Flow chains with forward/backward evaluation
- Log-det Jacobian tracking
- Custom ChainRules adjoints
- Conditioning on auxiliary parameters
- Flux-compatible trainable layers
- Efficient forward! in-place transforms
- Tools for building complex invertible models

## 📦 Installation

DensityFlows.jl is registered in the Julia **General registry**.

To install the latest released version, simply run:

```julia
using Pkg
Pkg.add("DensityFlows")
```
To use the latest development version:
```julia
Pkg.add(url="https://github.com/gaetanfacchinetti/DensityFlows.jl")
```

## 📘 Documentation

**A detailed documentation with a public API can be found [here](https://gaetanfacchinetti.github.io/docs/DensityFlows.jl/).**

Source code is documentation is also viewable via Julia’s help system:
```julia
using DensityFlows

?FlowChain
?Flow
?train!
```

## ⚡ Quick example

For $x$ an array of $d$-dimensional sampled data points and $\theta$ an array of the associated $n$-dimensional conditions, the following code shows how to prepare the data, create a normalizing flow model, train it and use it.

```julia
using DensityFlows

# package the data in a DataArrays object
data = DataArrays(x, θ)

# create the model assuming d=5
chain = FlowChain(
    CouplingLayer(data, [1, 2, 3], hidden_dim_s=16,  hidden_dim_t=16), 
    CouplingLayer(data, [3, 4, 5], hidden_dim_s=16,  hidden_dim_t=16), 
    CouplingLayer(data, [5, 1, 2], hidden_dim_s=16,  hidden_dim_t=16), 
    NormalizationLayer(x, -1f0, 1f0)
    )

# create and print a summary of the flow
@summary flow = Flow(chain, data)

# train the model over 50 epochs
state = Optimisers.setup(Optimisers.Adam(1f-3), flow.model)
train!(flow, data, state, epochs=50)

# sample new data points (assuming n=2) 
# for condition θ = [-1, 3] and θ = [2, sqrt(2)]
x_new_1 = sample(flow, 1000, (-1f0, 3f0))
x_new_2 = sample(flow, 1000, (2f0, sqrt(2)))
```

## 🤝 Contributing

Any feedback or suggestions to improve the code are very welcome! If you find a bug, have an idea for a new feature, or think something could be cleaner, feel free to open an issue or submit a pull request.

For developers:
- Fork the repository
- Clone your fork
    ```bash
    git clone https://github.com/<your-username>/DensityFlows.jl.git
    ```
- Activate development mode
    ```julia
    using Pkg
    Pkg.develop(path = "<path-to-your-fork>/DensityFlows.jl")
    ```

## 📝 License

DensityFlows.jl is released under the GPLv3 license.
See the [LICENSE](LICENSE) file for details.

## Related packages

For more advanced tools, check out [NormalizingFlows.jl](https://github.com/TuringLang/NormalizingFlows.jl).