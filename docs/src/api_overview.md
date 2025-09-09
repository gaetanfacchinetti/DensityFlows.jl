```@meta
CollapsedDocStrings = true
```


# Overview


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


## Evaluation

```@docs
backward
forward
```


## Save and load

```@docs
save
load
```