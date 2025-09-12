```@meta
CollapsedDocStrings = true
```


# Couplings


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
AffineCouplingElement
AffineCouplingLayer
AffineCouplingBlock
RNVPCouplingLayer
```

## Specific functions

```@docs
RNVP_backward
```




