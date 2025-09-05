##################################################################################
# This file is part of DensityFlows.jl
#
# Copyright (c) 2025, Gaétan Facchinetti
#
# DensityFlows.jl is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. DensityFlows.jl is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
##################################################################################
#
# Contains functions related to affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

import Base: reverse

export AffineCouplingElement, AffineCouplingLayer, RNVPCouplingLayer, NICECouplingLayer
export AffineCouplingBlock, AffineCouplingChain, AffineCouplingAxes
export reverse, forward, backward, forward!



##################################################################################
##################################################################################
# METHODS

@doc raw"""

    AffineCouplingAxes(d, n=0; kws...)

Create axes for AffineCouplingLayer.

# Arguments
- `d::Int`: dimension of the flow.
- `n::Int`: number of conditions / parameters (default is 0).
- `j::Int`: dimension cut (default is `d`÷2).
- `reverse::Bool`: (default is `false`).

The dimension cut `j` specifies which dimensions are not modified by the layer. 
If `reverse` is `false` the layer acts as the identity on dimensions (1, `j`).
If `reverse` is `true`  the layer acts as the identity on dimensions (`j`+1, `d`).
"""
function AffineCouplingAxes(
    d::Int,
    n::Int = 0;
    j::Int = d ÷ 2,
    reverse::Bool = false
    )

    # create symmetric blocks by default j = d ÷ 2 
    # with unchanged variables at the bottom

    axis_id = !reverse ? UnitRange(1, j)   : UnitRange(j+1, d) # dimensions on which we apply Identity
    axis_af = !reverse ? UnitRange(j+1, d) : UnitRange(1, j) # dimensions on which we apply Affine transformation

    axis_nn = vcat(UnitRange(1, n), axis_id .+ n)

    return AffineCouplingAxes(d, n, axis_id, axis_af, axis_nn, reverse)

end



@doc raw"""
    
    Base.reverse(axes)

Swap the dimensions that are left unchanged by the layer.
See also [`AffineCouplingAxes`](@ref).
"""
function Base.reverse(axes::AffineCouplingAxes)
    
    # exchange axis_id and axis_af
    axis_nn = vcat(UnitRange(1, axes.n), axes.axis_af .+ axes.n)
    return AffineCouplingAxes(axes.d, axes.n, axes.axis_af, axes.axis_id, axis_nn, !axes.reverse)

end



function AffineCouplingLayer(;
    axes::AffineCouplingAxes,
    input_dim::Int,
    output_dim::Int,
    hidden_dim::Int = 32,
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 0)

    t_sublayers = (
        [Flux.Dense(input_dim, hidden_dim, Flux.relu)],
        [Flux.Dense(hidden_dim, hidden_dim, Flux.relu) for _ in 1:n_sublayers_t],
        [Flux.Dense(hidden_dim, output_dim)]
    )
    
    t_net = Flux.Chain(vcat(t_sublayers...)...)

    if n_sublayers_s <= 0
        return NICECouplingLayer(t_net, axes)
    end

    s_sublayers = (
        [Flux.Dense(input_dim, hidden_dim, Flux.relu)],
        [Flux.Dense(hidden_dim, hidden_dim, Flux.relu) for _ in 1:n_sublayers_s],
        [Flux.Dense(hidden_dim, output_dim)]
    )

    s_net = Flux.Chain(vcat(s_sublayers...)...)

    return RNVPCouplingLayer(s_net, t_net, axes)

end


function AffineCouplingLayer(
    axes::AffineCouplingAxes;
    hidden_dim::Int = 32,
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 0)

    input_dim  = length(axes.axis_nn)
    output_dim = length(axes.axis_af)

    return AffineCouplingLayer(
        axes = axes,
        input_dim = input_dim, 
        output_dim = output_dim, 
        hidden_dim = hidden_dim, 
        n_sublayers_s = n_sublayers_s, 
        n_sublayers_t = n_sublayers_t)

end


@doc raw"""
    
    AffineCouplingLayer(axes; kws...)
    AffineCouplingLayer(d, n=0; kws...)
      
Create an AffineCouplingLayer with NN models `s` and `t`.

The layer can be represented as a function `f`
such that on dimensions where it does not act 
like the identity it returns

```math
    f(x) = x * \exp(s) + t \quad {\rm if~forward}
```

and

```math
    f^{-1}(z) = \exp(-s) * (z-t) \quad {\rm if~backward} \, .
```

# Arguments
- `axes::AffineCouplingAxes`.
- `d::Int`: dimension of the flow.
- `n::Int`: number of conditions / parameters (default is 0).
- `j::Int`: dimension cut (default is `d`÷2).
- `reverse::Bool`: (default is `false`).
- `hidden_dim::Int`: number of hidden dimensions in `s` and `t` (default is 32).
- `n_sublayers_t::Int`: number of sublayers in `t` (default is 2).
- `n_sublayers_s::Int`: number of sublayers in `s` (default is 2).

If `axes` is provided, then `d`, `n`, `j` and `reverse` should not be passed as arguments
as they would be redondant. 

The dimension cut `j` specifies which dimensions are not modified by the layer. 
If `reverse` is `false` the layer acts as the identity on dimensions (1, `j`).
If `reverse` is `true`  the layer acts as the identity on dimensions (`j`+1, `d`).

See also [`AffineCouplingAxes`](@ref).
"""
function AffineCouplingLayer(
    d::Int,
    n::Int = 0;
    j::Int = d ÷ 2,
    reverse::Bool = false,
    hidden_dim::Int = 32,
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 0)

    axes = AffineCouplingAxes(d, n, j = j, reverse = reverse)
    
    return AffineCouplingLayer(
        axes, 
        hidden_dim = hidden_dim, 
        n_sublayers_s = n_sublayers_s, 
        n_sublayers_t = n_sublayers_t)

end


function AffineCouplingBlock(;
    axes_1::AffineCouplingAxes,
    axes_2::AffineCouplingAxes,
    input_dim_1::Int,
    output_dim_1::Int,
    input_dim_2::Int,
    output_dim_2::Int,
    hidden_dim::Int = 32,
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 0)

    flow_1 = AffineCouplingLayer(
        axes = axes_1,
        input_dim = input_dim_1, 
        output_dim = output_dim_1, 
        hidden_dim = hidden_dim, 
        n_sublayers_s = n_sublayers_s, 
        n_sublayers_t = n_sublayers_t)

    flow_2 = AffineCouplingLayer(
        axes = axes_2,
        input_dim = input_dim_2, 
        output_dim = output_dim_2, 
        hidden_dim = hidden_dim, 
        n_sublayers_s = n_sublayers_s, 
        n_sublayers_t = n_sublayers_t)
    
    return AffineCouplingBlock(flow_1, flow_2)
    
end



@doc raw"""
    
    AffineCouplingBlock(axes; kws...)
    
Create an block of two [`AffineCouplingLayer`](@ref) with opposite axes.

Opposite axes here means that one is set with `reverse` = `true` 
and the other with `reverse` = `false`.

# Arguments
- `axes::AffineCouplingAxes`.
- `hidden_dim::Int`: number of hidden dimensions in `s` and `t` (default is 32).
- `n_sublayers_t::Int`: number of sublayers in `t` (default is 2).
- `n_sublayers_s::Int`: number of sublayers in `s` (default is 2).
"""
function AffineCouplingBlock(
    axes::AffineCouplingAxes;
    hidden_dim::Int = 32,
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 0)

    axes_2 = reverse(axes)

    input_dim_1  = length(axes.axis_nn)
    output_dim_1 = length(axes.axis_af)
    input_dim_2  = length(axes_2.axis_nn)
    output_dim_2 = length(axes_2.axis_af)

    return AffineCouplingBlock(
        axes_1 = axes,
        axes_2 = axes_2,
        input_dim_1 = input_dim_1, 
        output_dim_1 = output_dim_1, 
        input_dim_2 = input_dim_2, 
        output_dim_2 = output_dim_2, 
        hidden_dim = hidden_dim, 
        n_sublayers_s = n_sublayers_s, 
        n_sublayers_t = n_sublayers_t)

end


@doc raw"""
    
    AffineCouplingChain(n_couplings, axes, U; kws...)
    AffineCouplingChain(xs...)
    
Create an chain of [`AffineCouplingLayer`](@ref) or [`AffineCouplingBlock`](@ref).

Can either create a chain of `n_couplings` similar layers or blocks by passing
`n_couplings` and `axes` or instantiate a chain from pre-existings layers or 
blocks passed as `xs`.

# Arguments
- `n_couplings::Int`: number of couplings.
- `axes::AffineCouplingAxes`.
- `U:Type`: type of struct in the chain, can be `AffineCouplingLayer` or `AffineCouplingBlock` (default is `AffineCouplingBlock`).

Keywords arguments `kws...` are passed to the constructor of `AffineCouplingLayer` or `AffineCouplingBlock`.
"""
function AffineCouplingChain(
    n_couplings::Int, 
    axes::AffineCouplingAxes,
    ::Type{U} = AffineCouplingBlock;
    kws...
    ) where  {U<:AffineCouplingElement}

    stack = [U(axes; kws...) for _ in 1:n_couplings]
    
    return AffineCouplingChain(stack)
end


AffineCouplingChain(xs...) = AffineCouplingChain(xs)


##########################################################
## EVALUATION FUNCTIONS


@doc raw"""
    
    backward(f, x, θ=nothing)

Return ``f^{-1}(x \,|\, \theta)`` and ``J_{f^{-1}}(x \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `x::AbstractArray{T, N}`: arguments to pass to the flow element.
- `θ::Union{AbstractArray{T, N}, Nothing}`: parameters / conditions (default is `nothing`).
"""
backward


@doc raw"""
    
    forward(f, x, θ=nothing)

Return ``f(z  \,| \,\theta)`` and ``J_{f}(z \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `z::AbstractArray{T, N}`: arguments to pass to the flow element.
- `θ::Union{AbstractArray{T, N}, Nothing}`: parameters / conditions (default is `nothing`).
"""
forward


function backward(
    block::AffineCouplingBlock, 
    x::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    y, ln_det_jac_2 = backward(block.layer_2, x, θ)
    z, ln_det_jac_1 = backward(block.layer_1, y, θ)

    return z, (ln_det_jac_1 .+ ln_det_jac_2)
end


function forward(
    block::AffineCouplingBlock, 
    z::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    y, ln_det_jac_1 = forward(block.layer_1, z, θ)
    x, ln_det_jac_2 = forward(block.layer_2, y, θ)
 
    return x, (ln_det_jac_1 .+ ln_det_jac_2)
end

function forward!(
    block::AffineCouplingBlock, 
    z::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    forward!(block.layer_1, z, θ)
    forward!(block.layer_2, z, θ)
end


function backward(
    chain::AffineCouplingChain, 
    x::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    n = length(chain.layers)
    x_i , ln_det_jac = backward(chain.layers[end], x, θ)

    @inbounds for i ∈ 2:n
        x_i, ln_det_jac_i = backward(chain.layers[n - i + 1], x_i, θ)
        ln_det_jac = ln_det_jac .+ ln_det_jac_i
    end

    return x_i, ln_det_jac

end



function forward(
    chain::AffineCouplingChain, 
    z::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    n = length(chain.layers)
    z_i, ln_det_jac = forward(chain.layers[1], z, θ)

    @inbounds for i ∈ 2:n
        z_i, ln_det_jac_i = forward(chain.layers[i], z_i, θ)
        ln_det_jac = ln_det_jac .+ ln_det_jac_i
    end

    return z_i, ln_det_jac

end


function forward!(
    chain::AffineCouplingChain, 
    z::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    )  where {T<:AbstractFloat, N}

    for i ∈ eachindex(chain.layers)
        forward!(chain.layers[i], z, θ)
    end

end


