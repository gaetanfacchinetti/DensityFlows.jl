# Manual

## Quick start guide



## Example

```@repl
using DensityFlows

x = rand(2, 5000);

data = DataArrays(x)
flow = AffineCouplingFlow(3, data.metadata, n_sublayers_s=2, n_sublayers_t=2)
```

