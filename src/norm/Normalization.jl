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
# Contains functions related to normalization layers
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export NormalizationElement, NormalizationLayer

abstract type NormalizationElement <: FlowElement end

struct NormalizationLayer{T, U<:AbstractVector{T}} <: NormalizationElement
    x_min::U
    x_max::U
    α::T
    β::T
end

function NormalizationLayer(x::AbstractArray{T, N}, α::T = T(0), β::T = T(1)) where {T, N}
    x_min = vec(minimum(x, dims=2:N))
    x_max = vec(maximum(x, dims=2:N))

    @assert β > α "Bounds of the normalisation need to be in the correct order, β > α."
    return NormalizationLayer(x_min, x_max, α, β)
end

Flux.@layer NormalizationLayer trainable=()
@auto_functor NormalizationLayer

function DensityFlows.backward(
    nlayer::NormalizationLayer, 
    x::AbstractArray{T, N},
    θ::AbstractArray{T, N}
    ) where {T, N}

    x_diff = nlayer.x_max .- nlayer.x_min
    δ = nlayer.β - nlayer.α
    z = (nlayer.β .* (x .- nlayer.x_min)  +  nlayer.α .* (nlayer.x_max .- x) ) ./ x_diff
    ln_det_jac = - sum(log.(x_diff ./ δ)) .* ones(T, size(x)[2:N]...)
    # need to add the ones above in order to ensure type stability

    return z, ln_det_jac
end

function DensityFlows.forward(
    nlayer::NormalizationLayer, 
    z::AbstractArray{T, N},
    θ::AbstractArray{T, N}
    ) where {T, N}

    x_diff = nlayer.x_max .- nlayer.x_min
    δ = nlayer.β - nlayer.α
    x = (x_diff .* z .- nlayer.α .* nlayer.x_max  .+ nlayer.β .* nlayer.x_min) ./ δ
    ln_det_jac = +  sum(log.(x_diff ./ δ)) .* ones(T, size(z)[2:N]...)
    # need to add the ones above in order to ensure type stability

    return x, ln_det_jac
end


function DensityFlows.forward!(
    nlayer::NormalizationLayer, 
    z::AbstractArray{T},
    θ::AbstractArray{T}
    ) where {T}

    z .= ((nlayer.x_max .- nlayer.x_min) .* z .- nlayer.α .* nlayer.x_max  .+ nlayer.β .* nlayer.x_min) ./ (nlayer.β - nlayer.α)
    return nothing
end


function summarize(obj::NormalizationLayer)
    println("Normalization Layer")
end
