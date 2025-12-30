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

struct NICECouplingLayer{T, D, N, CT<:FluxChain} <: ConditionalFlowElement{T,D,N}
    
    t_net::CT
    axes::CouplingAxes
    
end


@doc raw"""
    
    NICECouplingLayer(axes; kws...)
    NICECouplingLayer(d, j = d ÷ 2; n=0, reverse=false, kws...)
    NICECouplingLayer(d, mask; n=0, kws...)
    
    NICECouplingLayer(t_net, axes)
      
Create an NICECouplingLayer with NN models `t`.

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
- `hidden_dim::Int`: number of hidden dimensions `t` (default is 32).
- `n_sublayers_t::Int`: number of sublayers in `t` (default is 2).
- `σ::Function`: activation function (default is `Flux.relu`).
- `bias::Bool`: activate bias (default is `true`).

# Example
```jldoctest
julia> @summary NICECouplingLayer(3, [1, 3], n=2, hidden_dim=10, n_sublayers_s=1, σ=Flux.tanh)
NICECouplingLayer | t_net > [3, 10, 10, 2] (172 parameters)
                  | axes  > (d,n)=(3,2); identity=(2), transformed=(1,3)
```

See also [`CouplingAxes`](@ref) and [`RNVPCouplingLayer`](@ref).
"""
function NICECouplingLayer end 

function NICECouplingLayer(s_net::FluxChain{T}, t_net::FluxChain{T}, axes::CouplingAxes) where {T}
    return NICECouplingLayer{T, axes.d, axes.n, typeof(s_net), typeof(t_net)}(s_net, t_net, axes)
end


function NICECouplingLayer(
    axes::CouplingAxes;
    n_sublayers_t::Int = 2, 
    hidden_dim_t::Int = 32,
    σ_t::Function = Flux.relu,
    kws...
    )
 
    input_dim  = length(axes.axis_nn)
    output_dim = length(axes.axis_af)

    t_net = _dflt_net(input_dim, output_dim, n_sublayers_t; hidden_dim = hidden_dim_t, σ = σ_t, kws...)    

    return NICECouplingLayer(t_net, axes)

end



function NICECouplingLayer{D, N}(j::Int = D ÷ 2; reverse::Bool = false, kws...) where {D, N} 
    return NICECouplingLayer(CouplingAxes(D, j, n=N, reverse=reverse); kws...)
end

function NICECouplingLayer{T, D, N}(j::Int = D ÷ 2; reverse::Bool = false, kws...) where {T, D, N} 
    @assert T === Float32 "By default NICECoupling layers are defined for Float32"
    return NICECouplingLayer(CouplingAxes(D, j, n=N, reverse=reverse); kws...)
end

function NICECouplingLayer{D, N}(mask::AbstractVector{<:Integer}; kws...) where {D, N} 
    return NICECouplingLayer(CouplingAxes(D, mask, n=N); kws...)
end

function NICECouplingLayer{T, D, N}(mask::AbstractVector{<:Integer}; kws...) where {T, D, N} 
    @assert T === Float32 "By default NICECoupling layers are defined for Float32"
    return NICECouplingLayer(CouplingAxes(D, mask, n=N); kws...)
end

#NICECouplingLayer(d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) = NICECouplingLayer(CouplingAxes(d, j, n=n, reverse=reverse); kws...)
#NICECouplingLayer(d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) = NICECouplingLayer(CouplingAxes(d, mask, n=n); kws...)



Flux.@layer NICECouplingLayer trainable=(t_net,)


# create a functor function to call RNVPCouplingLayer(...) 
# in place of forward(::RNVPCouplingLayer, ...) if wanted
@auto_functor NICECouplingLayer


# Define a custom show function
function summarize(obj::NICECouplingLayer)

    dim_t_net = [size(obj.t_net.net.layers[1].weight, 2), [size(l.weight, 1) for l in obj.t_net.net.layers]...]

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
    t::AbstractArray{T,K}, 
    u::AbstractArray{T}, 
    axis_id::AbstractVector{Int},
    axis_af::AbstractVector{Int}
    ) where {T,K}

    # Compute log-det-Jacobian
    ln_det_jac = zeros(T, size(t)[2:K]...) 

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
    layer::NICECouplingLayer{T}, 
    x::AbstractArray{T,K}, 
    θ::AbstractArray{T,K}
    ) where {T,K}

    # define the input from the
    input = selectdim(vcat(θ, x), 1, layer.axes.axis_nn)
    
    # get the output from the s and t neural networks
    t = layer.t_net(input)

    return NICE_backward(t, x, layer.axes.axis_id, layer.axes.axis_af)

end


function forward(
    layer::NICECouplingLayer{T}, 
    z::AbstractArray{T,K}, 
    θ::AbstractArray{T,K}
    ) where {T,K}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    t = layer.t_net(input)

    # ln|det J[T^{-1}]|
    ln_det_jac = zeros(T, size(t)[2:K]...) 

    x = similar(z)
    selectdim(x, 1, layer.axes.axis_id) .= selectdim(z, 1, layer.axes.axis_id)
    selectdim(x, 1, layer.axes.axis_af) .= selectdim(z, 1, layer.axes.axis_af) .+ t 

    return x, ln_det_jac
end


function forward!(
    layer::NICECouplingLayer, 
    z::AbstractArray{T,K}, 
    θ::AbstractArray{T,K}
    ) where {T,K}

    input = selectdim(vcat(θ, z), 1, layer.axes.axis_nn)

    t = layer.t_net(input)

    z_af = selectdim(z, 1, layer.axes.axis_af)
    z_af .= z_af .+ t

    return nothing
end

