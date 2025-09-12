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

export reverse, forward, backward, forward!



##################################################################################
##################################################################################
# METHODS


##################################################################################
# Axes

@doc raw"""

    AffineCouplingAxes(d, mask; kws )
    AffineCouplingAxes(d, j=d÷2; kws...)
    
Create axes for AffineCouplingLayer.

# Arguments
- `d::Int`: dimension of the flow.
- `j::Int`: dimension cut (default is `d`÷2).
- `mask::AbstractVector{Int}`: dimensions that are affected by the coupling.
- `n::Int`: number of conditions / parameters (default is 0).
- `reverse::Bool`: (default is `false`).

The dimension cut `j` specifies which dimensions are not modified by the layer. 
If `reverse` is `false` the layer acts as the identity on dimensions (1, `j`).
If `reverse` is `true`  the layer acts as the identity on dimensions (`j`+1, `d`).
"""
function AffineCouplingAxes(
    d::Int,
    mask::AbstractVector{Int};
    n::Int = 0
    )

    @assert maximum(mask) <= d "The mask cannot contain values higher than the dimension"

    # dimensions on which we apply Identity
    axis_id = findall(x -> !(x in mask), range(1, d))

    # dimensions on which we apply Affine transformation
    axis_af = mask

    # dimensions passed to the NN, n first for the parameters
    # and then in input gives all the unmodified (Identity) dimensions
    # (because of the triangular nature of the decomposition
    # the affine transformed dimensions only depend on the 
    # unmodified ones)
    axis_nn = vcat(UnitRange(1, n), axis_id .+ n)

    return AffineCouplingAxes(d, n, axis_id, axis_af, axis_nn)

end

function AffineCouplingAxes(
    d::Int,
    j::Int = d ÷ 2;
    n::Int = 0,
    reverse::Bool = false
    )

    mask = !reverse ? UnitRange(j+1, d) : UnitRange(1, j) 
    return AffineCouplingAxes(d, mask, n=n)

end



@doc raw"""
    
    Base.reverse(axes)

Swap the dimensions that are left unchanged by the layer.
See also [`AffineCouplingAxes`](@ref).
"""
function Base.reverse(axes::AffineCouplingAxes)
    
    # exchange axis_id and axis_af
    axis_nn = vcat(UnitRange(1, axes.n), axes.axis_af .+ axes.n)
    return AffineCouplingAxes(axes.d, axes.n, axes.axis_af, axes.axis_id, axis_nn)

end

function is_reverse(axes_1::AffineCouplingAxes, axes_2::AffineCouplingAxes)
    return all(axes_1.axis_af .== axes_2.axis_id) && all(axes_2.axis_af .== axes_1.axis_id) && (axes_1.n == axes_2.n)
end



##################################################################################
# Layers


function _dflt_net(
    input_dim::Int,
    output_dim::Int,
    n::Int;
    hidden_dim::Int = 32,
    σ::Function = Flux.relu,
    bias::Bool = true)

        
    sublayers = (
        [Dense(input_dim, hidden_dim, σ, bias = bias)],
        [Dense(hidden_dim, hidden_dim, σ, bias = bias) for _ in 1:(n-1)],
        [Dense(hidden_dim, output_dim, Flux.identity, bias = bias)]
    )
    
    return Flux.Chain(vcat(sublayers...)...)

end



@doc raw"""
    
    AffineCouplingLayer([T=RNVPCouplingLayer, ] axes; kws...)
    AffineCouplingLayer([T=RNVPCouplingLayer, ] d, j = d ÷ 2; n=0, reverse=false, kws...)
    AffineCouplingLayer([T=RNVPCouplingLayer, ] d, mask; n=0, kws...)
    
    AffineCouplingLayer(t_net, axes)
    AffineCouplingLayer(s_net, t_net, axes)
      
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

By default `s` and `t` are built with `Dense` neural networks.

# Arguments
- `axes::AffineCouplingAxes`.
- `d::Int`: dimension of the flow.
- `j::Int`: dimension cut (default is `d`÷2).
- `mask::Vector{Int}`: dimensions that are affected by the coupling.

# Keywords arguments
- `n::Int`: number of conditions / parameters (default is 0).
- `hidden_dim::Int`: number of hidden dimensions in `s` and `t` (default is 32).
- `n_sublayers_t::Int`: number of sublayers in `t` (default is 2).
- `n_sublayers_s::Int`: number of sublayers in `s` (default is 2).
- `σ::Function`: activation function (default is `Flux.relu`).
- `bias::Bool`: activate bias (default is `true`).

# Example
```jldoctest
julia> @summary AffineCouplingLayer(3, [1, 3], n=2, hidden_dim=10, n_sublayers_s=1, σ=Flux.tanh)
• RNVPCouplingLayer > s_net: [3, 10, 2] (62 parameters)
• RNVPCouplingLayer > t_net: [3, 10, 10, 2] (172 parameters)
• RNVPCouplingLayer > axes: (d,n)=(3,2), id=[2], af=[1, 3]
```

See also [`AffineCouplingAxes`](@ref).
"""
function AffineCouplingLayer end


@doc raw"""
    
    AffineCouplingBlock(layer_1, layer_2)
    AffineCouplingBlock([T=RNVPCouplingLayer, ] first_axes; kws...)
    AffineCouplingBlock([T=RNVPCouplingLayer, ] d, j = d ÷ 2; n=0, reverse=false, kws...)
    AffineCouplingBlock([T=RNVPCouplingLayer, ] d, mask; n=0, kws...)
    
Create an block of two [`AffineCouplingLayer`](@ref) with opposite / complementary axes.

# Arguments
- `first_axes::AffineCouplingAxes`: axes of the first layer.
- `d::Int`: dimension of the flow.
- `j::Int`: dimension cut (default is `d`÷2).
- `mask::Vector{Int}`: dimensions that are affected by the coupling.

# Keyword arguments
- `hidden_dim::Int`: number of hidden dimensions in `s` and `t` (default is 32).
- `n_sublayers_t::Int`: number of sublayers in `t` (default is 2).
- `n_sublayers_s::Int`: number of sublayers in `s` (default is 2).
- `σ::Function`: activation function (default is `Flux.relu`).
- `bias::Bool`: activate bias (default is `true`).).

# Example
```jldoctest
julia> @summary AffineCouplingBlock(3, [1, 3], n=2, hidden_dim=10, n_sublayers_s=1, σ=Flux.tanh)
• RNVPCouplingLayer > s_net: [3, 10, 2] (62 parameters)
• RNVPCouplingLayer > t_net: [3, 10, 10, 2] (172 parameters)
• RNVPCouplingLayer > axes: (d,n)=(3,2), id=[2], af=[1, 3]
• RNVPCouplingLayer > s_net: [4, 10, 1] (61 parameters)
• RNVPCouplingLayer > t_net: [4, 10, 10, 1] (171 parameters)
• RNVPCouplingLayer > axes: (d,n)=(3,2), id=[1, 3], af=[2]
```
"""
function AffineCouplingBlock end


AffineCouplingLayer(t_net::Flux.Chain, axes::AffineCouplingAxes) = NICECouplingLayer(t_net, axes)
AffineCouplingLayer(s_net::Flux.Chain, t_net::Flux.Chain, axes::AffineCouplingAxes) = RNVPCouplingLayer(s_net, t_net, axes)


for fname in (:AffineCouplingLayer, :AffineCouplingBlock)
    @eval begin
        $fname(axes::AffineCouplingAxes; kws...) = $fname(RNVPCouplingLayer, axes; kws...)
        $fname(d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) = $fname(AffineCouplingAxes(d, j, n=n, reverse=reverse); kws...)
        $fname(d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) = $fname(AffineCouplingAxes(d, mask, n=n); kws...)
        $fname(::Type{T}, d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) where {T<:AffineCouplingLayer} = $fname(AffineCouplingAxes(d, j, n=n, reverse=reverse); kws...)
        $fname(::Type{T}, d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) where {T<:AffineCouplingLayer} = $fname(AffineCouplingAxes(d, mask, n=n); kws...)
    end 
end


function AffineCouplingLayer(
    ::Type{T},
    axes::AffineCouplingAxes;
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 2,
    kws...
    ) where {T<:AffineCouplingLayer}
 

    input_dim  = length(axes.axis_nn)
    output_dim = length(axes.axis_af)

    t_net = _dflt_net(input_dim, output_dim, n_sublayers_t; kws...)
    (T === NICECouplingLayer) && return T(t_net, axes)
    
    s_net = _dflt_net(input_dim, output_dim, n_sublayers_s; kws...)

    return T(s_net, t_net, axes)

end



function AffineCouplingBlock(::Type{T}, first_axes::AffineCouplingAxes; kws...) where {T<:AffineCouplingLayer}
 
    # define a second axis opposite / complementary to the first one
    second_axes = reverse(first_axes)

    layer_1 = AffineCouplingLayer(T, first_axes; kws...)
    layer_2 = AffineCouplingLayer(T, second_axes; kws...)

    return AffineCouplingBlock(layer_1, layer_2)

end



##########################################################
## EVALUATION FUNCTIONS


function backward(
    block::AffineCouplingBlock, 
    x::AbstractArray{T, N},
    θ::AbstractArray{T, N} = dflt_θ(x)
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




