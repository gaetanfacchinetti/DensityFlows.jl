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
# Contains functions related to Blocks of layers
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export CouplingBlock

@doc raw"""
    
    CouplingBlock(layer_1, layer_2)
    CouplingBlock([T=RNVPCouplingLayer, ] first_axes; kws...)
    CouplingBlock([T=RNVPCouplingLayer, ] d, j = d ÷ 2; n=0, reverse=false, kws...)
    CouplingBlock([T=RNVPCouplingLayer, ] d, mask; n=0, kws...)
    
Create an block of two [`CouplingLayer`](@ref) with opposite / complementary axes.

# Arguments
- `first_axes::CouplingAxes`: axes of the first layer.
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
julia> @summary CouplingBlock(3, [1, 3], n=2, hidden_dim=10, n_sublayers_s=1, σ=Flux.tanh)
RNVPCouplingLayer | s_net > [3, 10, 2] (62 parameters)
                  | t_net > [3, 10, 10, 2] (172 parameters)
                  | axes  > (d,n)=(3,2); identity=(2), transformed=(1,3)
RNVPCouplingLayer | s_net > [4, 10, 1] (61 parameters)
                  | t_net > [4, 10, 10, 1] (171 parameters)
                  | axes  > (d,n)=(3,2); identity=(1,3), transformed=(2)
```
"""
CouplingBlock


struct CouplingBlock{T<:CouplingLayer, U<:CouplingLayer} <: FlowElement

    layer_1::T
    layer_2::U

    # inner constructor to ensure that the layers are complementary
    function CouplingBlock(layer_1::T, layer_2::U) where {T<:CouplingLayer, U<:CouplingLayer}
        !(is_reverse(layer_1.axes, layer_2.axes)) && throw(ArgumentError("layer_1 and layer_2 need to have complementary axes"))
        return new{T, U}(layer_1, layer_2)
    end

end

Flux.@layer CouplingBlock
@auto_functor CouplingBlock

Base.length(obj::CouplingBlock) = 2

function summarize(obj::CouplingBlock)
    summarize(obj.layer_1)
    summarize(obj.layer_2)
end


function CouplingBlock(
    ::Type{T}, 
    first_axes::CouplingAxes; 
    kws...
    ) where {T<:CouplingLayer}
 
    # define a second axis opposite / complementary to the first one
    second_axes = reverse(first_axes)

    layer_1 = CouplingLayer(T, first_axes; kws...)
    layer_2 = CouplingLayer(T, second_axes; kws...)

    return CouplingBlock(layer_1, layer_2)

end

CouplingBlock(axes::CouplingAxes; kws...) = CouplingBlock(RNVPCouplingLayer, axes; kws...)
CouplingBlock(d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) = CouplingBlock(CouplingAxes(d, j, n=n, reverse=reverse); kws...)
CouplingBlock(d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) = CouplingBlock(CouplingAxes(d, mask, n=n); kws...)
CouplingBlock(::Type{T}, d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) where {T<:CouplingLayer} = CouplingBlock(CouplingAxes(d, j, n=n, reverse=reverse); kws...)
CouplingBlock(::Type{T}, d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) where {T<:CouplingLayer} = CouplingBlock(CouplingAxes(d, mask, n=n); kws...)


##########################################################
## EVALUATION FUNCTIONS

function backward(
    block::CouplingBlock, 
    x::AbstractArray{T,N},
    θ::AbstractArray{T,N}
    ) where {T,N}

    y, ln_det_jac_2 = backward(block.layer_2, x, θ)
    z, ln_det_jac_1 = backward(block.layer_1, y, θ)

    return z, (ln_det_jac_1 .+ ln_det_jac_2)
end


function forward(
    block::CouplingBlock, 
    z::AbstractArray{T,N},
    θ::AbstractArray{T,N}
    ) where {T,N}

    y, ln_det_jac_1 = forward(block.layer_1, z, θ)
    x, ln_det_jac_2 = forward(block.layer_2, y, θ)
 
    return x, (ln_det_jac_1 .+ ln_det_jac_2)
end


function forward!(
    block::CouplingBlock, 
    z::AbstractArray{T,N},
    θ::AbstractArray{T,N}
    ) where {T,N}

    forward!(block.layer_1, z, θ)
    forward!(block.layer_2, z, θ)
end



