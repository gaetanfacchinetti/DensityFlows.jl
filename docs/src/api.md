```@meta
CollapsedDocStrings = true
```


# Public API



## Structure
```@docs
Flow
FlowElement
AffineCouplingElement
RNVPCouplingLayer
```

The hierarchy of types can be visualised from the following outputs  

```@repl
using DensityFlows
AffineCouplingElement <: FlowElement
AffineCouplingLayer <: AffineCouplingElement
RNVPCouplingLayer <: AffineCouplingLayer
```

## Axes

Axes define and manipulate dimensions on which the [`AffineCouplingElement`](@ref) operates. Some dimensions are left unchanged while the others undergo an affine transformation.

```@docs
AffineCouplingAxes
```

Axes can be manipulated with the following functions.

```@docs
Base.reverse
```


## Coupling elements

```@docs
AffineCouplingLayer
AffineCouplingBlock
```


## Chains

```@docs
AffineCouplingChain
```


## Evaluation

```@docs
backward
forward
```

## Save and loads

```@docs
save
load
```


## Data

```@docs
DataArrays
MetaData
DataPartition
```

```@docs
data_x_min
data_x_max
data_θ_min
data_θ_max
normalize_input
normalize_input!
resize_output
resize_output!
```