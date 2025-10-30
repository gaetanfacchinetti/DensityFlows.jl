# Example

Let us assume that we have the following data

```@example 1; continued=true
n = 10000

# parameter that is fixed to -1 or 2
θ = Matrix{Float32}(undef, (1, 2*n))
@views θ[1, 1:n] .= -1f0
@views θ[1, (n+1):end] .= 2f0

# 3D data
x1 = randn(2*n)
x2 = sin.(x1/1.1) .+ 0.3*randn(2*n) .+ θ[1, :]
x3 = exp.(x1/1.4)/10 .+  0.1*θ[1, :] .*randn(2*n) .- 0.1 * θ[1, :]

x = Float32.(vcat(x1', x2', x3'))
```
![](images/distrib.svg)

We can first define the data as

```@example 1; continued=true
using DensityFlows

data = DataArrays(x, θ)
ax   = CouplingAxes(3, n=1)
```

Or we can also directly use a coupling Layer

```@example 1

chain = FlowChain(
    CouplingBlock(ax, n_sublayers_s=2, n_sublayers_t=2), 
    CouplingBlock(ax, n_sublayers_s=2, n_sublayers_t=2), 
    CouplingBlock(ax, n_sublayers_s=2, n_sublayers_t=2),
    NormalizationLayer(x, -1f0, 1f0)
    )

@summary flow = Flow(chain, data)
```