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
# Contains functions related to the chains of flows
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


######################## 
## Documentation

@doc raw"""
    
    backward(f, x, θ=nothing)

Return ``f^{-1}(x \,|\, \theta)`` and ``J_{f^{-1}}(x \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `x::AbstractArray{T, N}`: arguments to pass to the flow element.
- `θ::Union{AbstractArray{T, N}, Nothing}`: parameters / conditions (default is `nothing`).
"""
backward


@doc raw"""
    
    forward(f, z, θ=nothing)

Return ``f(z  \,| \,\theta)`` and ``J_{f}(z \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `z::AbstractArray{T, N}`: arguments to pass to the flow element.
- `θ::Union{AbstractArray{T, N}, Nothing}`: parameters / conditions (default is `nothing`).
"""
forward


@doc raw"""
    
    forward!(f, z, θ=nothing)

Replace `z` by ``f(z  \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `z::AbstractArray{T, N}`: arguments to pass to the flow element.
- `θ::Union{AbstractArray{T, N}, Nothing}`: parameters / conditions (default is `nothing`).
"""
forward!


######################## 

export FlowChain

struct FlowChain{T<:Union{Tuple, AbstractVector}} <: FlowElement
    layers::T
end

FlowChain(xs...) = FlowChain(xs)

@auto_flow FlowChain

function Base.show(io::IO, obj::FlowChain)
    for layer in obj.layers
        show(io, layer)
    end
end


@doc raw"""
    
    FlowChain(n_couplings, axes, U; kws...)
    FlowChain(xs...)
    
Create an chain of FlowElements.

Can either create a chain of `n_couplings` similar layers or blocks by passing
`n_couplings` and `axes` or instantiate a chain from pre-existings layers or 
blocks passed as `xs`.

# Arguments
- `n_couplings::Int`: number of couplings.
- `axes::AffineCouplingAxes`.
- `U:Type`: type of struct in the chain, can be `AffineCouplingLayer` or `AffineCouplingBlock` (default is `AffineCouplingBlock`).

Keywords arguments `kws...` are passed to the constructor of `AffineCouplingLayer` or `AffineCouplingBlock`.
See also [`AffineCouplingLayer`](@ref) or [`AffineCouplingBlock`](@ref).
"""
function FlowChain(
    n_couplings::Int, 
    axes::AffineCouplingAxes,
    ::Type{U} = AffineCouplingBlock;
    kws...
    ) where  {U<:AffineCouplingElement}

    stack = [U(axes; kws...) for _ in 1:n_couplings]
    
    return FlowChain(stack)
end



function backward(
    chain::FlowChain, 
    x::AbstractArray{T, N},
    θ::AbstractArray{T, N} = dflt_θ(x)
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
    chain::FlowChain, 
    z::AbstractArray{T, N},
    θ::AbstractArray{T, N} = dflt_θ(z)
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
    chain::FlowChain, 
    z::AbstractArray{T, N},
    θ::AbstractArray{T, N} = dflt_θ(z)
    )  where {T<:AbstractFloat, N}

    for i ∈ eachindex(chain.layers)
        forward!(chain.layers[i], z, θ)
    end

end




