```@meta
CollapsedDocStrings = true
```


# Overview


## Structure

```@docs
Flow
FlowElement
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
forward!
```


## Save and load

```@docs
save
load
```

## Macros

```@docs
@auto_flow
@auto_forward
@auto_functor
```