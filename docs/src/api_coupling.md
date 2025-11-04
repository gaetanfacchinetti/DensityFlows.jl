```@meta
CollapsedDocStrings = true
```


# Couplings


## Axes

Axes define and manipulate dimensions on which the [`CouplingLayer`](@ref) operates. Some dimensions are left unchanged while the others undergo an affine transformation.

```@docs
CouplingAxes
```

Axes can be manipulated with the following functions.

```@docs
Base.reverse
```


## Coupling elements

```@docs
CouplingLayer
CouplingBlock
RNVPCouplingLayer
```

## Specific functions

```@docs
RNVP_backward
NICE_backward
```




