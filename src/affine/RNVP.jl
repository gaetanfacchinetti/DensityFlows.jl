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
# Contains functions related to real-NVP affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


export RNVPCouplingLayer, RNVP_backward

############
# Real-NVP layer structure

@doc raw"""

    RNVPCouplingLayer(s_net, t_net, axes)

Structure of a Real-Non-Volume-Preserving affine coupling layer.

Contains 
"""
struct RNVPCouplingLayer{T<:Flux.Chain, U<:Flux.Chain} <: CouplingLayer
    
    s_net::T
    t_net::U

    axes::CouplingAxes

end

# make the coupling layer parameters trainable
Flux.@layer RNVPCouplingLayer trainable=(s_net, t_net)

# create a functor function to call RNVPCouplingLayer(...) 
# in place of forward(::RNVPCouplingLayer, ...) if wanted
@auto_functor RNVPCouplingLayer


# Define a custom show function
function summarize(obj::RNVPCouplingLayer)

    dim_s_net = [size(obj.s_net.layers[1].weight, 2), [size(l.weight, 1) for l in obj.s_net.layers]...]
    dim_t_net = [size(obj.t_net.layers[1].weight, 2), [size(l.weight, 1) for l in obj.t_net.layers]...]

    println("RNVPCouplingLayer | s_net > $dim_s_net ($(sum(length, Flux.trainables(obj.s_net))) parameters)")
    println("                  | t_net > $dim_t_net ($(sum(length, Flux.trainables(obj.t_net))) parameters)")
    print("                  | axes  > ")
    summarize(obj.axes)
    println("")
end

@doc raw"""

    RNVP_backward(s, t, u, axis_id, axis_af)

Return z = (u-t)*exp(s), ln(det|J|) = sum(s).
"""
function RNVP_backward(
    s::AbstractArray{T}, 
    t::AbstractArray{T}, 
    u::AbstractArray{T}, 
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int}
    ) where {T}

    # Compute log-det-Jacobian
    ln_det_jac = - dropdims(sum(s, dims = 1), dims = 1)

    # because the operation below is can not be treated automatically
    # differentiated by Zygote, we write our own rrule below
    z = similar(u)
    selectdim(z, 1, axis_id) .= selectdim(u, 1, axis_id)
    selectdim(z, 1, axis_af) .= (selectdim(u, 1, axis_af) .- t) .* exp.(-s)

    return z, ln_det_jac

end


function ChainRulesCore.rrule(
    ::typeof(RNVP_backward),
    s::AbstractArray{T}, 
    t::AbstractArray{T}, 
    u::AbstractArray{T},
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int} 
    ) where {T}

    # Evaluate the function
    z, ln_det_jac = RNVP_backward(s, t, u, axis_id, axis_af)

    # Let us call R the output of the entire NN
    # z̄ = ∂R/∂z, s̄ = ∂R/∂s, etc...
    # then we need to return s̄, t̄, ū from z̄, j̄ (where j = ln_det_jac)
    # From the chain rule
    # s̄ = ∂R/∂s = ∂R/∂z * ∂z/∂s + ∂R/∂j * ∂j/∂s
    # t̄ = ∂R/∂t = ∂R/∂z * ∂z/∂t + ∂R/∂j * ∂j/∂t
    # ū = ∂R/∂u = ∂R/∂z * ∂z/∂u + ∂R/∂j * ∂j/∂u
    # as z = (u-t)*exp(-s) and j = sum(s) this yields the relations below
    function RNVP_backward_pullback(ȳ)
        
        z̄, j̄ = ȳ

        z̄_af = selectdim(z̄, 1, axis_af)
        z̄_id = selectdim(z̄, 1, axis_id)
        
        u_af = selectdim(u, 1, axis_af)

        # create copies of log_det_jac matching size of s
        shaped_j̄ = similar(s, size(s))
        @inbounds for i in 1:size(s, 1)
            selectdim(shaped_j̄, 1, i) .= j̄
        end

        s̄ = - z̄_af .* (u_af .- t) .* exp.(-s) .- shaped_j̄
        t̄ = - z̄_af .* exp.(-s)
        
        ū = zeros(T, size(u))
        selectdim(ū, 1, axis_af) .= z̄_af .* exp.(-s)
        selectdim(ū, 1, axis_id) .= z̄_id

        return ChainRulesCore.NoTangent(), s̄, t̄, ū, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()

    end 

    return (z, ln_det_jac), RNVP_backward_pullback

end


function backward(
    layer::RNVPCouplingLayer, 
    x::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,N}

    # define the input from the
    input = selectdim(vcat(θ, x), 1, layer.axes.axis_nn)
    
    # get the output from the s and t neural networks
    s = layer.s_net(input)
    t = layer.t_net(input)

    return RNVP_backward(s, t, x, layer.axes.axis_id, layer.axes.axis_af)

end


function forward(
    layer::RNVPCouplingLayer, 
    z::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,N}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    s = layer.s_net(input)
    t = layer.t_net(input)

    # ln|det J[T^{-1}]|
    ln_det_jac = dropdims(sum(s, dims = 1), dims = 1)

    x = similar(z)
    selectdim(x, 1, layer.axes.axis_id) .= selectdim(z, 1, layer.axes.axis_id)
    selectdim(x, 1, layer.axes.axis_af) .= selectdim(z, 1, layer.axes.axis_af) .* exp.(s) .+ t 

    return x, ln_det_jac
end


function forward!(
    layer::RNVPCouplingLayer, 
    z::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,N}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    s = layer.s_net(input)
    t = layer.t_net(input)

    z_af = selectdim(z, 1, layer.axes.axis_af)
    z_af .= z_af .* exp.(s) .+ t

    return nothing
end


