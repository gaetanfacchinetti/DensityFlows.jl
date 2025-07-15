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

export AffineCouplingInstance, AffineCouplingLayer, RNVPCouplingLayer, NICECouplingLayer
export AffineCouplingBlock, AffineCouplingChain, AffineCouplingAxes
export reverse, forward, backward, forward!


##################################################################################
##################################################################################
# TYPES

############
# Affine axes structure

struct AffineCouplingAxes
    
    d::Int # total number of dimensions without the conditions
    n::Int # number of conditions
    
    axis_id::Vector{Int}
    axis_af::Vector{Int}
    axis_nn::Vector{Int}

    reverse::Bool
    
end

abstract type AffineCouplingInstance <: FlowInstance end 
abstract type AffineCouplingLayer <: AffineCouplingInstance end

############
# Real-NVP layer structure

struct RNVPCouplingLayer{T<:Flux.Chain, U<:Flux.Chain} <: AffineCouplingLayer
    
    s_net::T
    t_net::U

    axes::AffineCouplingAxes

end

Flux.@layer RNVPCouplingLayer
Functors.@functor RNVPCouplingLayer

# Specify that axes are not in the trainable parameters
Optimisers.trainable(m::RNVPCouplingLayer) = (;s_net = Optimisers.trainable(m.s_net), t_net = Optimisers.trainable(m.t_net))


############
# NICE layer structure

struct NICECouplingLayer{CT<:Flux.Chain} <: AffineCouplingLayer
    
    t_net::CT  
    axes::AffineCouplingAxes
    
end


Flux.@layer NICECouplingLayer
Functors.@functor NICECouplingLayer

# Specify that axes are not in the trainable parameters
Optimisers.trainable(m::NICECouplingLayer) = (;t_net = Optimisers.trainable(m.t_net))


############
# Affine block layer structure

struct AffineCouplingBlock{T<:AffineCouplingLayer, U<:AffineCouplingLayer} <: AffineCouplingInstance
    layer_1::T
    layer_2::U
end

Flux.@layer AffineCouplingBlock
Functors.@functor AffineCouplingBlock


############
# Affine chain structure

struct AffineCouplingChain{T<:Union{Tuple, AbstractVector}} <: AffineCouplingInstance
    layers::T
end

Flux.@layer AffineCouplingChain
Functors.@functor AffineCouplingChain






##################################################################################
##################################################################################
# METHODS


function AffineCouplingAxes(
    d::Int,
    n::Int = 0;
    j::Int = d ÷ 2,
    reverse::Bool = false
    )

    # create symmetric blocks by default j = d ÷ 2 
    # with unchanged variables at the bottom

    axis_id = !reverse ? UnitRange(1, j)   : UnitRange(j+1, d)
    axis_af = !reverse ? UnitRange(j+1, d) : UnitRange(1, j)

    axis_nn = vcat(UnitRange(1, n), axis_id .+ n)

    return AffineCouplingAxes(d, n, axis_id, axis_af, axis_nn, reverse)

end



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



function AffineCouplingLayer(
    d::Int,
    n::Int = 0,
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


function AffineCouplingChain(
    n_couplings::Int, 
    axes::AffineCouplingAxes,
    ::Type{U} = AffineCouplingBlock;
    kws...
    ) where  {U<:AffineCouplingInstance}

    stack = [U(axes; kws...) for _ in 1:n_couplings]
    
    return AffineCouplingChain(stack)
end


AffineCouplingChain(xs...) = AffineCouplingChain(xs)


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

    forward!(forward!(block.layer_1, z, θ), z, θ)
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