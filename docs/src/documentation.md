# Documentation

Let us assume a ``d``-dimensional dataset organised as an `Array{Float32}` called `x` such that `size(x, 1) = d`. Further assume n parameters / conditions gathered into a `Array{Float32}` called `θ` such that `size(θ, 1) = n`. To each `x` value should correspond one `θ` value, i.e `size(x)[2:end] == size(θ)[2:end]`. If these conditions are satisfied one can find the distribution of ``x`` knowing ``\theta`` from the following steps.

## Prepare the data

For convinivence the data can be stored in a `DataArrays` object as shown below. For the purpose of this quick start guide we use dummy normal distributed variables. See [the example section](./example.md) for a more realistic scenario. 

```@example guide
using Flux
using Optimisers
using LinearAlgebra
using Distributions
using DensityFlows

x = randn(Float32, 7, 100)
θ = randn(Float32, 2, 100)

data = DataArrays(x, θ, f_training = 0.8, f_validation=0.2)
```

The sum of the training and validation fractions must be below or equal to 1. They are used to randomly partition the data before training.

## Elementary layers

The elementary elements of the flow are the [`CouplingLayers`](./api_coupling.md). A coupling layer is more particularly a bijective transformation which acts on some dimensions while leaving the others unchanged. In order to perform a transformation of all dimensions a chain of coupling layers is necessary. All definitions of coupling layer below are equivalent and transform dimensions 4, 5, 6, and 7.
```@example guide; continued=true
CouplingLayer(data)
CouplingLayer(data, 3)
CouplingLayer(data, [4, 5, 6, 7])
CouplingLayer(7, [4, 5, 6, 7], n=2)
CouplingLayer(7, 3, n=2)
```
and gives
```@example guide
@summary CouplingLayer(7, 3, n=2)
```
One can also use the `reverse` option to rather transform the dimensions 1, 2, and 3
```@example guide; continued=true
CouplingLayer(data, reverse=true)
CouplingLayer(data, 3, reverse=true)
CouplingLayer(data, [1, 2, 3])
CouplingLayer(7, [1, 2, 3], n=2)
CouplingLayer(7, 3, n=2, reverse=true)
```
in which case
```@example guide
@summary CouplingLayer(7, 3, n=2, reverse=true)
```

Layers can then be stacked in a [`FlowChain`](@ref) to create a flow. An example would be
```@example guide
chain = FlowChain(
    CouplingLayer(data, [4, 5, 6, 7]), 
    CouplingLayer(data, [2, 3, 4, 5]), 
    CouplingLayer(data, [7, 1, 2, 3]), 
    NormalizationLayer(x, -1f0, 1f0)
    )
@summary chain
```
where good practice is to place a [`NormalizationLayer`](@ref) at the end to increase the performances of the network and avoid `NaN`s.

!!! warning "Mixing dimensions"
    When using RNVP-layers, as implemented by default, the chain should at least contain several layers with the _transformed_ dimensions shuffled from one layer to the other, in order for all dimensions to be _transformed_ at lease once.

!!! tip "One dimensional distribution"
    RNVP-layers are _diagonal_ in that they only _transform_ one part of the dimensions while taking the other part in input. Emulating a 1 dimensional distribution is thus not straightforward. Emulating a one dimensional distribution with RNVP-layers is nonetheless possible by artificially promoting the distribution to two dimensions, associating any point to a value drawn from a known distribution like a Gaussian distribution.

## Blocks: combination of layers

A `CouplingBlock` is a combination of two layers, with complentary transformations on axes. That is, if the first layer transforms dimensions 1, 3, 4, 7 the second transforms dimensions 2, 5, and 6 by construction. Said differently, the following chains are equivalent

```@example guide
chain = FlowChain(
    CouplingLayer(data, [4, 5, 6, 7]), 
    CouplingLayer(data, [1, 2, 3]), 
    )
@summary chain
```

```@example guide
chain = FlowChain(
    CouplingBlock(data, [4, 5, 6, 7]), 
    )
@summary chain
```

## More about the layers

By default coupling layers are set as Real-NVP which requires two neural networks called `s` for scaling and `t` for translation. Both of these networks are instances of `Flux.Dense` but their properties can also be modified from the [`CouplingLayer`](@ref) constructor. First, the following two declarations are equivalent.
```@example guide
@summary CouplingLayer(data)
```
```@example guide
@summary CouplingLayer(RNVPCouplingLayer, data)
```
Second, properties of the `s` and `t` network can be changed as follows.
```@example guide
@summary CouplingLayer(data, n_sublayers_s=3, n_sublayers_t=4, hidden_dim_s=16, hidden_dim_t=12, σ_s=Flux.relu, σ_t=Flux.sigmoid)
```
Finally then can also be set direclty but one then needs to be carefull with the input and output dimensions. The input dimension should be `number of untransformed dimensions` + ``n`` and the output dimension should be `number of transformed dimensions`.
```@example guide
s_net = Flux.Chain([Flux.Dense(5, 32, Flux.sigmoid), Flux.Dense(32, 16, Flux.relu), Flux.Dense(16, 4)])
t_net = Flux.Chain([Flux.Dense(5, 12, Flux.relu), Flux.Dense(12, 16, Flux.logcosh), Flux.Dense(16, 32, Flux.relu), Flux.Dense(32, 4)])
@summary CouplingLayer(s_net, t_net, data)
```

## Train the model

First define the flow from the a chain of layers and a base distribution. By default the latter is set to a multivariate Normal distribution but any distribution from the `Distributions` package can be used.
```julia
flow = Flow(chain, data)

# which is equivalent to
base = Distributions.MvNormal(zeros(Float32, 7), LinearAlgebra.diagm(ones(FLoat32, 7)))
flow = Flow(base, chain, data)
```
Then, one needs to define the state of the model and the optimiser.
```julia
state = Optimisers.setup(Optimisers.Adam(1f-3), flow.model)
```
Finally one can call the implemented [`train!`](@ref) function.
```julia
train!(flow, data, state, epochs=100, batchsize=64)
```

## Save and load the model

After training the model can be saved.
```julia
# if directory "my_flow" does not already exists
@save_flow "my_flow" flow

# to overwrite any existing directory / model with the same name
@clear_and_save_flow "my_flow" flow 
```
Similarly it can also be loaded back.
```julia
flow = @load_flow "my_flow"
```


## Use the model

The model can be used to sample new data points or to extract the probability distribution function. To sample a ``(d, r, s)``, with ``(r, s)\in \mathbb{N}_*^2`` array of an unconditional flow simply one calls the function [`sample`](@ref).
```julia
sample(flow, (r, s))
```
If the flow is conditional, values for those conditions ``\theta`` need to be provided. There are two possible ways, either you can specify a different value of ``\theta`` for each drawn value or give a single value for the entire sample. In the first case one must define `θ::AbstractArray{T, k}` where ``k=3`` (in this example) and of size ``(n, r, s)``. In the second case, one can introduce the condition as a tuple of size ``n``, `θ::NTuple{n, T}`.  One then calls the same function [`sample`](@ref).
```julia
sample(flow, (r, s), θ)
```
The model can also be used to directly extract the value of the probability distribution function. For a ``d``-dimensional flow, define `NTuple{d, Vector{T}}` where every entry is a vector of values where the pdf must be evaluated. For instance, in the case of an unconditional ``3``-dimensional flow, one can compute the pdf on points ``(x=2, y=3, z=1)``, ``(x=2, y=2, z=1)``, ``(x=2, y=3, z=4)`` from the following call to [`pdf`](@ref).
```julia
res = pdf(flow, ([2], [3, 2], [1, 4]))

# res[1, 1, 1] = pdf(2, 3, 1)
# res[1, 1, 2] = pdf(2, 3, 4)
# res[1, 2, 1] = pdf(2, 2, 1)
# res[1, 2, 2] = pdf(2, 2, 4)
```
For a conditional flow, as for sampling case, the conditions can be passed as a tuple of size ``n``, `θ::NTuple{n, T}`.
```julia
res = pdf(flow, ([2], [3, 2], [1, 4], θ)
```

!!! info "logpdf"
    The natural logarithm of the probability distribution funtion ``\ln p`` can also be obtained similarly calling [`logpdf`](@ref).

## To go further: define custom layers

The code has been built such that it is easy to implement new layers. It must take the form of a struct on which is applied the `Flux.@layer` macro specifying the trainable parameters. Then it must define a `forward` and `backward` functions that are differentiable by `Zygote` (or use a custom chain rule). These two functions must have signature `(::NewLayer, ::AbstractArray{T,N}, ::AbstractArray{T,N}) where {T,N}` and return the transformed array as well as the logarithm of the determinant of the jacobian of the transformation.

```@example guide
struct NewLayer{U, V} <: FlowElement
    a::U
    b::V
end

# make the coupling layer parameters trainable
Flux.@layer NewLayer trainable=(a,)

# dummy forward and backward functions
forward(layer::NewLayer, z::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = a .* z .+ b, a
backward(layer::NewLayer, x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = (x .- b) ./ a, one(T)/a
```

One must then also define a `forward!` function. One can define a custom one or use a macro to define a default. Similarly using another macro one can also define a functor in place of `forward(...)` to call `layer(...)` but this is optional.
```@example guide
@auto_forward! NewLayer
@auto_functor NewLayer
```

By default such a structure is saved as two files, one for `a` and one for `b` if both are simple elements everything can be saved in a single file using the macro
```@example guide
@save_as_atomic NewLayer
```


