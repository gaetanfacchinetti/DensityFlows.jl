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
# Based on L. Dinh, J. Sohl-Dickstein, S. Bengio (2017). Density Estimation Using Real NVP. International Conference on Learning Representations. 
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


export RNVPCouplingLayer, RNVP_backward

const GRAD_CLAMPING = Ref{Float32}(-1f0)

############
# Real-NVP layer structure


struct RNVPCouplingLayer{T, D, N, CS<:FluxChain{T}, CT<:FluxChain{T}} <: CouplingLayer{T,D,N}
    
    s_net::CS
    t_net::CT

    axes::CouplingAxes

end

@doc raw"""
    
    RNVPCouplingLayer(axes; kws...)
    RNVPCouplingLayer(d, j = d ÷ 2; n=0, reverse=false, kws...)
    RNVPCouplingLayer(d, mask; n=0, kws...)

    RNVPCouplingLayer(s_net, t_net, axes)
      
Create an RNVPCouplingLayer with NN models `s` and `t`.

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
- `axes::CouplingAxes`.
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
julia> @summary RNVPCouplingLayer(3, [1, 3], n=2, hidden_dim=10, n_sublayers_s=1, σ=Flux.tanh)
RNVPCouplingLayer | s_net > [3, 10, 2] (62 parameters)
                  | t_net > [3, 10, 10, 2] (172 parameters)
                  | axes  > (d,n)=(3,2); identity=(2), transformed=(1,3)
```

See also [`CouplingAxes`](@ref) and and [`NICECouplingLayer`](@ref).
"""
function RNVPCouplingLayer end 

function RNVPCouplingLayer(s_net::FluxChain{T}, t_net::FluxChain{T}, axes::CouplingAxes) where {T}
    return RNVPCouplingLayer{T, axes.d, axes.n, typeof(s_net), typeof(t_net)}(s_net, t_net, axes)
end

function _dflt_net(
    input_dim::Int,
    output_dim::Int,
    n::Int;
    hidden_dim::Int = 32,
    σ::Function = Flux.relu,
    bias::Bool = true)

        
    sublayers = (
        [Flux.Dense(input_dim, hidden_dim, σ, bias = bias)],
        [Flux.Dense(hidden_dim, hidden_dim, σ, bias = bias) for _ in 1:(n-1)],
        [Flux.Dense(hidden_dim, output_dim, Flux.identity, bias = bias)]
    )
    
    return FluxChain(vcat(sublayers...)...)

end

function RNVPCouplingLayer(
    axes::CouplingAxes;
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 2,
    hidden_dim_t::Int = 32,
    hidden_dim_s::Int = 32,
    σ_t::Function = Flux.relu,
    σ_s::Function = Flux.relu,
    kws...
    )
 
    input_dim  = length(axes.axis_nn)
    output_dim = length(axes.axis_af)

    t_net = _dflt_net(input_dim, output_dim, n_sublayers_t; hidden_dim = hidden_dim_t, σ = σ_t, kws...)    
    s_net = _dflt_net(input_dim, output_dim, n_sublayers_s; hidden_dim = hidden_dim_s, σ = σ_s, kws...)

    return RNVPCouplingLayer(s_net, t_net, axes)

end


RNVPCouplingLayer(d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) = RNVPCouplingLayer(CouplingAxes(d, j, n=n, reverse=reverse); kws...)
RNVPCouplingLayer(d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) = RNVPCouplingLayer(CouplingAxes(d, mask, n=n); kws...)

# make the coupling layer parameters trainable
Flux.@layer RNVPCouplingLayer trainable=(s_net, t_net)

# create a functor function to call RNVPCouplingLayer(...) 
# in place of forward(::RNVPCouplingLayer, ...) if wanted
@auto_functor RNVPCouplingLayer


# Define a custom show function
function summarize(obj::RNVPCouplingLayer)

    dim_s_net = [size(obj.s_net.net.layers[1].weight, 2), [size(l.weight, 1) for l in obj.s_net.net.layers]...]
    dim_t_net = [size(obj.t_net.net.layers[1].weight, 2), [size(l.weight, 1) for l in obj.t_net.net.layers]...]

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
    s::AbstractArray{T,K}, 
    t::AbstractArray{T,K}, 
    u::AbstractArray{T,K}, 
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int}
    ) where {T,K}

    # Compute log-det-Jacobian
    ln_det_jac = - reshape(sum(s, dims = 1), size(s)[2:K])

    # because the operation below cannot be automatically
    # differentiated by Zygote, we write our own rrule below
    z = similar(u)
    selectdim(z, 1, axis_id) .= selectdim(u, 1, axis_id)
    selectdim(z, 1, axis_af) .= (selectdim(u, 1, axis_af) .- t) .* exp.(-s)

    return z, ln_det_jac

end


function ChainRulesCore.rrule(
    ::typeof(RNVP_backward),
    s::AbstractArray{T,K}, 
    t::AbstractArray{T,K}, 
    u::AbstractArray{T,K},
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int} 
    ) where {T,K}

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

        # gradient clamping for better stability (if turned on)
        (GRAD_CLAMPING[] > 0) && (z̄ .= T.(GRAD_CLAMPING[] * tanh.(z̄ / GRAD_CLAMPING[])))
        
        #z̄, j̄ = clamp.(ȳ, -10.0, 10.0)

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
        
        ū = zero.(similar(u)) #zeros(T, size(u))
        selectdim(ū, 1, axis_af) .= z̄_af .* exp.(-s)
        selectdim(ū, 1, axis_id) .= z̄_id

        return ChainRulesCore.NoTangent(), s̄, t̄, ū, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()

    end 

    return (z, ln_det_jac), RNVP_backward_pullback

end


function backward(
    layer::RNVPCouplingLayer{T}, 
    x::AbstractArray{T,K}, 
    θ::AbstractArray{T,K}
    ) where {T,K}

    # define the input from the
    input = selectdim(vcat(θ, x), 1, layer.axes.axis_nn)
    
    # get the output from the s and t neural networks
    s = 5 * Flux.tanh.(layer.s_net(input)/5)
    t = layer.t_net(input)

    return RNVP_backward(s, t, x, layer.axes.axis_id, layer.axes.axis_af)

end


function forward(
    layer::RNVPCouplingLayer{T}, 
    z::AbstractArray{T,K}, 
    θ::AbstractArray{T,K}
    ) where {T,K}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    s = 5 * Flux.tanh.(layer.s_net(input)/5)
    t = layer.t_net(input)

    # ln|det J[T^{-1}]|
    ln_det_jac = dropdims(sum(s, dims = 1), dims = 1)

    x = similar(z)
    selectdim(x, 1, layer.axes.axis_id) .= selectdim(z, 1, layer.axes.axis_id)
    selectdim(x, 1, layer.axes.axis_af) .= selectdim(z, 1, layer.axes.axis_af) .* exp.(s) .+ t 

    return x, ln_det_jac
end


function forward!(
    layer::RNVPCouplingLayer{T}, 
    z::AbstractArray{T,K}, 
    θ::AbstractArray{T,K}
    ) where {T,K}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    s = 5 * Flux.tanh.(layer.s_net(input)/5)
    t = layer.t_net(input)

    z_af = selectdim(z, 1, layer.axes.axis_af)
    z_af .= z_af .* exp.(s) .+ t

    return nothing
end


