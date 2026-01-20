##################################################################################
# This file is part of DensityFlows.jl
#
# Copyright (c) 2025, Gaétan Facchinetti
#
# jl is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. jl is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
##################################################################################
#
# Contains functions related to transforming the condition input
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export Conditioner, transform_condition, NormalizationConditioner, ConditionerChain

abstract type Conditioner{T,P,N} end

function transform_condition(cond::Conditioner{T,P}, θ::AbstractArray{T}) where {T,P}
    @assert P == size(θ, 1)
    return cond(θ)
end

struct NormalizationConditioner{T, P, N, V<:AbstractVector{T}} <: Conditioner{T, P, N}
    θ_min::V
    θ_max::V
    α::T
    β::T
end

function NormalizationConditioner(
    θ_min::V,
    θ_max::V,
    α::T,
    β::T
    ) where {T, V<:AbstractVector{T}}

    return NormalizationConditioner{T, length(θ_min), length(θ_max), V}(θ_min, θ_max, α, β)
end

function NormalizationConditioner(θ::AbstractArray{T, K}, α::T = T(-1), β::T = T(1)) where {T, K}
    
    θ_min = vec(minimum(θ, dims=2:K))
    θ_max = vec(maximum(θ, dims=2:K))

    @assert β > α "Bounds of the normalisation need to be in the correct order, β > α."
    return NormalizationConditioner{T, size(θ, 1), size(θ, 1), typeof(θ_min)}(θ_min, θ_max, α, β)
end

NormalizationConditioner(data::DataArrays{T}, α::T = T(0), β::T = T(1)) where {T} = NormalizationConditioner(data.θ, α, β)

Flux.@layer NormalizationConditioner trainable=()

# Simple normalization
function (f::NormalizationConditioner{T})(θ::AbstractArray{T,K}) where {T,K}
    ((K >= 1) && (size(θ, 1) == 0)) && (return θ)
    return   (f.β .* (θ .- f.θ_min)  +  f.α .* (f.θ_max .- θ) ) ./ (f.θ_max .- f.θ_min)
end

number_parameters_input(cond::Conditioner{T, P, N}) where {T, P, N} = P
number_parameters_output(cond::Conditioner{T, P, N}) where {T, P, N} = N



# CHAINS OF CONDITIONERS
# SIMILAR TO FLOWCHAIN

struct ConditionerChain{T, P, N, C<:Union{Tuple{Vararg{<:Conditioner{T}}}, AbstractVector{<:Conditioner{T}}}} <: Conditioner{T,P,N}
    
    layers::C

    function ConditionerChain(
        xs::Union{Tuple{Vararg{<:Conditioner{T}}}, AbstractVector{<:Conditioner{T}}}
        ) where {T} 
    
        @assert length(xs) >= 1 

        # TO DO: ADD A CHECK THAT THE DIMENSIONS ARE COMPATIBLE
        # FROM ONE LAYER TO THE NEXT

        return new{T, number_parameters_input(xs[1]), number_parameters_output(xs[end]), typeof(xs)}(xs)
    
    end

end

ConditionerChain(xs::Conditioner...) = ConditionerChain(xs)
ConditionerChain(::Type{T}, n::Int, args...; kws...) where {T<:Conditioner} = ConditionerChain([T(args...; kws...) for _ ∈ 1:n]...)

function transform_condition(conds::ConditionerChain{T,P}, θ::AbstractArray{T}) where {T,P}
    
    @assert P == size(θ, 1)
    
    res = conds[1](θ)
    
    for cond in conds[2:end]
        res = cond(res)
    end

    return res
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

    @eval $fname(chain::ConditionerChain, args...; kws...) = $fname(chain.layers, args...; kws...)
    
end
