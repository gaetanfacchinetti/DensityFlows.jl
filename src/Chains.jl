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


export backward, forward, forward!
export FlowChain, FlowChainAffine, concatenate

######################## 
## Documentation

@doc raw"""
    
    backward(f, x [, θ=dflt_θ(x)])

Return ``f^{-1}(x \,|\, \theta)`` and ``J_{f^{-1}}(x \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `x::AbstractArray{T}`: arguments to pass to the flow element.
- `θ::AbstractArray{T}`: parameters / conditions (default is [`dflt_θ`](@ref)).
"""
function backward end


@doc raw"""
    
    forward(f, z [, θ=dflt_θ(z)])

Return ``f(z  \,| \,\theta)`` and ``J_{f}(z \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `z::AbstractArray{T}`: arguments to pass to the flow element.
- `θ::AbstractArray{T}`: parameters / conditions (default is [`dflt_θ`](@ref)).
"""
function forward end


@doc raw"""
    
    forward!(f, z [, θ=dflt_θ(z)])

Replace `z` by ``f(z  \,| \,\theta)`` where ``\theta`` is an array of parameters.


# Arguments
- `z::AbstractArray{T}`: arguments to pass to the flow element.
- `θ::AbstractArray{T}`: parameters / conditions (default is [`dflt_θ`](@ref)).
"""
function forward! end


######################## 



struct FlowChain{T, D, C<:Union{Tuple{Vararg{<:FlowElement{T,D}}}, AbstractVector{<:FlowElement{T,D}}}} <: FlowElement{T,D}
    
    layers::C

    function FlowChain(
        xs::Union{Tuple{Vararg{<:FlowElement{T,D}}}, AbstractVector{<:FlowElement{T,D}}}
        ) where {T,D} 
    
        @assert length(xs) >= 1 

        conditions = [number_conditions(x) for x in xs]

        # check that the number of conditions in the same for all FlowElements
        # passed in input to the chain, avoid compatibility issues

        n = -1
        init = false
        
        for condition in conditions
            
            if !init && condition > 0
                n = condition
                init = true
            end

            if condition > 0
                @assert condition == n "Number of conditions must be the same for all elements in the chain"
            end

        end 

        return new{T, number_dimensions(xs[1]), typeof(xs)}(xs)
    end

end


Flux.@layer FlowChain 
@auto_functor FlowChain


@doc raw"""

    FlowChain(elements::Tuple)
    FlowChain(elements...)
    FlowChain([T = CouplingBlock, ], n, args...; kwars... )

Instanciate a chain of flow elements from a Tuple.

Possible to directly pass the elements of a chain, or construct a
chain of `n` identical blocks of type `T`, with `args...` 
and `kws...` passed to the constructor of `T`.
"""
function FlowChain end



# get the number of condition of a chain
function number_conditions(element::FlowChain)
    conditions = [number_conditions(x) for x in element.layers]
    return maximum(conditions)
end


FlowChain(xs::FlowElement...) = FlowChain(xs)
FlowChain(::Type{T}, n::Int, args...; kws...) where {T<:FlowElement} = FlowChain([T(args...; kws...) for _ ∈ 1:n]...)
FlowChain(n::Int, args...; kws...) = FlowChain(CouplingBlock, n, args...; kws...)


@doc raw"""

    concatenate(x::FlowChain...)
    concatenate(x::FlowChain, y::FlowElement...)
    concatenate(x::Union{Tuple, FlowElement}, y::FlowChain...)

Make one `FlowChain` from multiple chains or adding flow elements.
"""
concatenate(x::FlowChain...) = concatenate(x)
concatenate(x::FlowChain, y::FlowElement...) = FlowChain(x.layers..., y...)
concatenate(x::Tuple{Vararg{FlowElement}}, y::FlowChain) = FlowChain(x..., y.layers...)
concatenate(x::FlowElement, y::FlowChain) = FlowChain(x, y.layers...)

function concatenate(x::Tuple{Vararg{FlowChain}})
    layers = []
    for y in x, l in y.layers
        push!(layers, l)
    end
    return FlowChain(layers)
end

for fname ∈ (
    :(Base.getindex), 
    :(Base.length), 
    :(Base.first), 
    :(Base.last), 
    :(Base.iterate), 
    :(Base.lastindex),
    :(Base.firstindex),
    :(Base.keys),
    )

    @eval $fname(chain::FlowChain, args...; kws...) = $fname(chain.layers, args...; kws...)
    
end


# making a nicer show function
function summarize(obj::FlowChain)
    for layer ∈ obj
        summarize(layer)
    end
end


function backward(
    chain::FlowChain{T}, 
    x::AbstractArray{T,K},
    θ::AbstractArray{T,K}
    ) where {T,K}


    n = length(chain)
    x_i, ln_det_jac = backward(chain[end], x, θ)

    @inbounds for i ∈ 2:n
        x_i, ln_det_jac_i = backward(chain[n - i + 1], x_i, θ)
        ln_det_jac = ln_det_jac .+ ln_det_jac_i
    end

    return x_i, ln_det_jac

end



function forward(
    chain::FlowChain{T}, 
    z::AbstractArray{T,K},
    θ::AbstractArray{T,K}
    ) where {T,K}

    n = length(chain)
    z_i, ln_det_jac = forward(chain[1], z, θ)

    @inbounds for i ∈ 2:n
        z_i, ln_det_jac_i = forward(chain[i], z_i, θ)
        ln_det_jac = ln_det_jac .+ ln_det_jac_i
    end

    return z_i, ln_det_jac

end


function forward!(
    chain::FlowChain{T}, 
    z::AbstractArray{T,K},
    θ::AbstractArray{T,K}
    )  where {T,K}

    @inbounds for i ∈ eachindex(chain.layers)
        forward!(chain[i], z, θ)
    end

end



