```@meta
CollapsedDocStrings = true
```


# Overview


## Structure

```@docs
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
save_flow
```

## Macros

```@docs
@auto_forward!
@auto_functor
@summary
@flow_wrapper
@unconditional_wrapper
@save_as_atomic
@clear_and_save_flow
@save_flow
@load_flow
```