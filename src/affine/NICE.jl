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
# Contains functions related to NICE affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export NICECouplingLayer, NICE_backward

############
# NICE layer structure

struct NICECouplingLayer{T<:Flux.Chain} <: CouplingLayer
    
    t_net::T  
    axes::CouplingAxes
    
end

Flux.@layer NICECouplingLayer trainable=(t_net,)


# create a functor function to call RNVPCouplingLayer(...) 
# in place of forward(::RNVPCouplingLayer, ...) if wanted
@auto_functor NICECouplingLayer


# Define a custom show function
function summarize(obj::NICECouplingLayer)

    dim_t_net = [size(obj.t_net.layers[1].weight, 2), [size(l.weight, 1) for l in obj.t_net.layers]...]

    println("NICECouplingLayer | t_net > $dim_t_net ($(sum(length, Flux.trainables(obj.t_net))) parameters)")
    print("                  | axes  > ")
    summarize(obj.axes)
    println("")
end

@doc raw"""

    NICE_backward(t, u, axis_id, axis_af)

Return z = (u-t), zeros(...).
"""
function NICE_backward(
    t::AbstractArray{T,N}, 
    u::AbstractArray{T}, 
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int}
    ) where {T,N}

    # Compute log-det-Jacobian
    ln_det_jac = zeros(T, size(t)[2:N]...) 

    # because the operation below is can not be treated automatically
    # differentiated by Zygote, we write our own rrule below
    z = similar(u)
    selectdim(z, 1, axis_id) .= selectdim(u, 1, axis_id)
    selectdim(z, 1, axis_af) .= selectdim(u, 1, axis_af) .- t

    return z, ln_det_jac

end


function ChainRulesCore.rrule(
    ::typeof(NICE_backward),
    t::AbstractArray{T}, 
    u::AbstractArray{T},
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int} 
    ) where {T}

    # Evaluate the function
    z, ln_det_jac = NICE_backward(t, u, axis_id, axis_af)

    # Let us call R the output of the entire NN
    # z̄ = ∂R/∂z, t̄ = ∂R/∂t, etc...
    # then we need to return t̄, ū from z̄
    # From the chain rule
    # t̄ = ∂R/∂t = ∂R/∂z * ∂z/∂t + ∂R/∂j * ∂j/∂t
    # ū = ∂R/∂u = ∂R/∂z * ∂z/∂u + ∂R/∂j * ∂j/∂u
    # as z = (u-t) and j = 0 this yields the relations below
    function NICE_backward_pullback(ȳ)
        
        z̄, _ = ȳ

        t̄ = - selectdim(z̄, 1, axis_af)
        ū = z̄

        return ChainRulesCore.NoTangent(), t̄, ū, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()

    end 

    return (z, ln_det_jac), NICE_backward_pullback

end


function backward(
    layer::NICECouplingLayer, 
    x::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,N}

    # define the input from the
    input = selectdim(vcat(θ, x), 1, layer.axes.axis_nn)
    
    # get the output from the s and t neural networks
    t = layer.t_net(input)

    return NICE_backward(t, x, layer.axes.axis_id, layer.axes.axis_af)

end


function forward(
    layer::NICECouplingLayer, 
    z::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,N}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    t = layer.t_net(input)

    # ln|det J[T^{-1}]|
    ln_det_jac = zeros(T, size(t)[2:N]...) 

    x = similar(z)
    selectdim(x, 1, layer.axes.axis_id) .= selectdim(z, 1, layer.axes.axis_id)
    selectdim(x, 1, layer.axes.axis_af) .= selectdim(z, 1, layer.axes.axis_af) .+ t 

    return x, ln_det_jac
end


function forward!(
    layer::NICECouplingLayer, 
    z::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,N}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    t = layer.t_net(input)

    z_af = selectdim(z, 1, layer.axes.axis_af)
    z_af .= z_af .+ t

    return nothing
end

