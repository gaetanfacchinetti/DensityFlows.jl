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
# Contains functions related to normalization layers
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export NormalizationElement, GlobalNormalizationLayer

abstract type NormalizationElement{T,D,N} <: FlowElement{T,D,N} end

struct GlobalNormalizationLayer{T, D, N, V<:AbstractVector{T}} <: NormalizationElement{T,D,N}
    x_min::V
    x_max::V
    α::T
    β::T
    n::Integer
end

@doc raw""" 

    GlobalNormalizationLayer(x, α, β)

Create an overall normalization layer for overall dataset.

# Arguments
- `x::Union{AbstractArray{T, N}, DataArrays{T, N}}`: data
- `α::T`: minimum value of the output
- `β::T`: maximum value of the output
"""

function GlobalNormalizationLayer(
    x_min::V, 
    x_max::V,
    α::T,
    β::T,
    n::Integer
    ) where {T, V<:AbstractVector{T}}

    return GlobalNormalizationLayer{T, length(x_min), n, V}(x_min, x_max, α, β, n)
end

function GlobalNormalizationLayer(x::AbstractArray{T, K}, α::T, β::T; n::Integer) where {T, K}
    
    x_min = vec(minimum(x, dims=2:K))
    x_max = vec(maximum(x, dims=2:K))

    @assert β > α "Bounds of the normalisation need to be in the correct order, β > α."
    return GlobalNormalizationLayer(x_min, x_max, α, β, n)
end

GlobalNormalizationLayer(data::DataArrays{T, N}, α::T = T(0), β::T = T(1)) where {T, N} = GlobalNormalizationLayer(data.x, α, β, n = size(data.θ, 1))

Flux.@layer GlobalNormalizationLayer trainable=()
@auto_functor GlobalNormalizationLayer

function backward(
    nlayer::GlobalNormalizationLayer, 
    x::AbstractArray{T, N},
    θ::AbstractArray{T, N}
    ) where {T, N}

    x_diff = nlayer.x_max .- nlayer.x_min
    δ = nlayer.β - nlayer.α
    z = (nlayer.β .* (x .- nlayer.x_min)  +  nlayer.α .* (nlayer.x_max .- x) ) ./ x_diff
    ln_det_jac = - sum(log.(x_diff ./ δ)) .* one.(similar(x, eltype(x), size(x)[2:N]))
    # need to add the ones above in order to ensure type stability

    return z, ln_det_jac
end

function forward(
    nlayer::GlobalNormalizationLayer, 
    z::AbstractArray{T, N},
    θ::AbstractArray{T, N}
    ) where {T, N}

    x_diff = nlayer.x_max .- nlayer.x_min
    δ = nlayer.β - nlayer.α
    x = (x_diff .* z .- nlayer.α .* nlayer.x_max  .+ nlayer.β .* nlayer.x_min) ./ δ
    ln_det_jac = +  sum(log.(x_diff ./ δ)) .* one.(similar(z, eltype(z), size(z)[2:N]))
    # need to add the ones above in order to ensure type stability

    return x, ln_det_jac
end


function forward!(
    nlayer::GlobalNormalizationLayer, 
    z::AbstractArray{T},
    θ::AbstractArray{T}
    ) where {T}

    z .= ((nlayer.x_max .- nlayer.x_min) .* z .- nlayer.α .* nlayer.x_max  .+ nlayer.β .* nlayer.x_min) ./ (nlayer.β - nlayer.α)
    return nothing
end


function summarize(obj::GlobalNormalizationLayer)
    println("GlobalNormalization Layer")
end
