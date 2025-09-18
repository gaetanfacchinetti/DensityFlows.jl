```@meta
CollapsedDocStrings = true
```


# Overview


## Structure

```@docs
Flow
FlowElement
summarize
```

The hierarchy of types can be visualised from the following outputs  

```@repl
using DensityFlows
CouplingLayer <: FlowElement
RNVPCouplingLayer <: CouplingLayer
```


## Evaluation

```@docs
backward
forward
forward!
```


## Save and load

```@docs
save_element
load_element
```

## Macros

```@docs
@auto_forward!
@auto_functor
@summary
@unconditional_wrapper
```