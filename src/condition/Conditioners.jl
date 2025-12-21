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

export Conditioner, transform_condition, NormalizationConditioner

abstract type Conditioner{T,N,L} end

function transform_condition(cond::Conditioner{T,N}, θ::AbstractArray{T}) where {T,N}
    @assert N == size(θ, 1)
    return cond(θ)
end

struct NormalizationConditioner{T, N, L, U<:AbstractVector{T}} <: Conditioner{T, N, L}
    θ_min::U
    θ_max::U
    α::T
    β::T
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
    return  (f.β .* (θ .- f.θ_min)  +  f.α .* (f.θ_max .- θ) ) ./ (f.β - f.α)
end