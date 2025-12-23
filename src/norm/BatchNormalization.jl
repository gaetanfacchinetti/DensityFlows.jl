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
# Contains functions related to BatchNormalizationLayer
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export BatchNormalizationLayer

mutable struct BatchNormalizationLayer{T,D,N,V<:AbstractVector{T}} <: NormalizationElement{T,D,N}
   
    const β::V
    const γ::V
    
    const λ::T
    const ϵ::T

    m::V
    v::V

    init::Bool

    n::Integer
end

Flux.@layer BatchNormalizationLayer trainable=(β, γ)

function BatchNormalizationLayer(
    β::V, 
    γ::V, 
    λ::T, 
    ϵ::T, 
    m::V, 
    v::V, 
    init::Bool, 
    n::Integer
    ) where {T, V<:AbstractVector{T}}

    return BatchNormalizationLayer{T, length(β), n, V}(β, γ, λ, ϵ, m, v, init, n)

end

function BatchNormalizationLayer(::Type{T}, d::Integer, n::Integer, λ::T, ϵ::T, init::Bool = false) where {T}
    β = zeros(T, d)
    γ = zeros(T, d)
    m = zeros(T, d)
    v = zeros(T, d)
    BatchNormalizationLayer{T, d, n, Vector{T}}(β, γ, λ, ϵ, m, v, init, n)
end

BatchNormalizationLayer(d::Integer; n::Integer, λ::Float32 = 0.1f0, ϵ::Float32 = 1f-5) = BatchNormalizationLayer(Float32, d, n, λ, ϵ, false)
BatchNormalizationLayer(data::DataArrays{Float32}; λ::Float32 = 0.1f0, ϵ::Float32 = 1f-5) = BatchNormalizationLayer(Float32, number_dimensions(data), number_conditions(data), λ, ϵ, false)


function backward(
    layer::BatchNormalizationLayer{T},
    x::AbstractArray{T,K},
    θ::AbstractArray{T,K}
    ) where {T,K}

    # mean and variance of the batch
    m = reshape(Statistics.mean(x, dims=2:K), size(x, 1))
    v = reshape(Statistics.var(x, dims=2:K), size(x, 1))
    
    if !(layer.init)
        layer.m = m
        layer.v = v
        layer.init = true
    else
        # update the average and variance
        layer.m = (1-layer.λ) * layer.m .+ layer.λ * m
        layer.v = (1-layer.λ) * layer.v .+ layer.λ * v
    end

    γ = 5*Flux.tanh.(layer.γ/5)

    frame = one.(similar(x, eltype(x), size(x)[2:K]...))

    ln_det_jac = frame .* sum(layer.γ .- log.(layer.v .+ layer.ϵ) ./ 2)
    z = (x .- layer.m) ./ sqrt.(layer.v .+ layer.ϵ)  .* exp.(γ) .+ layer.β
    
    return z, ln_det_jac

end

function forward(
    layer::BatchNormalizationLayer{T},
    z::AbstractArray{T,K},
    θ::AbstractArray{T,K}
    ) where {T,K}

    γ = 5*Flux.tanh.(layer.γ/5)
    x = (z .- layer.β) .* exp.(-layer.γ) .* sqrt.(layer.v .+ layer.ϵ) .+ layer.m
    ln_det_jac = fill(-sum(γ .- log.(layer.v .+ layer.ϵ) ./ 2), size(x)[2:K]...)
    
    return x, ln_det_jac

end

function forward!(
    layer::BatchNormalizationLayer{T},
    z::AbstractArray{T,K},
    θ::AbstractArray{T,K}
    ) where {T,K}

    γ = 5*Flux.tanh.(layer.γ/5)
    z .= (z .- layer.β) .* exp.(-γ) .* sqrt.(layer.v .+ layer.ϵ) .+ layer.m
    return nothing 

end