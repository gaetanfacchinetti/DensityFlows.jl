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
# Contains functions related to noise modeling
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export NoiseModel, NoiseGenerator, NoNoise, GaussianNoise, ResampledNoise
export noisy_data

abstract type NoiseModel{T} end
abstract type NoiseGenerator{T} <: NoiseModel{T} end

struct NoNoise{T} <: NoiseGenerator{T} end

NoNoise() = NoNoise{Float32}()

(f::NoNoise)(x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = (x, θ)

struct GaussianNoise{T} <: NoiseGenerator{T}
    μx::T
    σx::T
    μθ::T
    σθ::T
end

noisy_data(noise::NoiseGenerator{T}, x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = noise(x, θ)
noisy_data(x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = (x, θ)

# by default define a noise in the parameters
GaussianNoise() = GaussianNoise(0.f0, 0.f0, 0.f0, 0f0)
GaussianNoise(σ::T) where {T} = GaussianNoise(zeros(T, 3)..., σ)
GaussianNoise(μ::T, σ::T) where {T} = GaussianNoise(zeros(T, 2)..., μ, σ)

function (f::GaussianNoise{T})(x::AbstractArray{T}, θ::AbstractArray{T}) where {T}
    return (x .+ muladd.(f.σx, randn(T, size(x)...), f.μx),  (θ .+ muladd.(f.σθ, randn(T, size(θ)...), f.μθ)))
end

struct ResampledNoise{T<:AbstractFloat, A<:AbstractMatrix{T}} <: NoiseGenerator{T}
    noise_x::A
    noise_θ::A
end

# by default define a noise in the parameters
ResampledNoise() = ResampledNoise(zeros(Float32, 1, 1), zeros(Float32, 1, 1)) 
ResampledNoise(noise::AbstractMatrix{T}) where {T} = ResampledNoise(zeros(T, 1, 1), noise) 

function (f::ResampledNoise{T})(x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} 
    rx = shuffle!(collect(1:size(f.noise_x, 2)))
    rθ = shuffle!(collect(1:size(f.noise_θ, 2)))
    return (x .+ f.noise_x[:, rx, ntuple(_ -> :, N-2)...], θ .+ f.noise_θ[:, rθ, ntuple(_ -> :, N-2)...])
end


struct NoiseTuple{N,T} <: NoiseModel{T}
    layers::NTuple{N, NoiseGenerator{T}}
end

NoiseTuple(xs::NoiseGenerator{T}...) where {T} = NoiseTuple(xs)
NoiseTuple(xs::AbstractVector{NoiseGenerator{T}}) where {T} = NoiseTuple(xs...)

function generate_noisy_data(
    noise::NoiseTuple{M,T}, 
    x::AbstractArray{T,N}, 
    θ::AbstractArray{T,N}
    ) where {T,M,N}

    y, z = noisy_data(noise.layers[M], x, θ)

    for j in (M-1):-1:1
        y, z = noisy_data(noise.layers[j], y, z)
    end

    return y, z

end

+(x::NoiseGenerator{T}, y::NoiseGenerator{T}) where {T} = NoiseTuple(x, y)
+(x::NoiseTuple{T}, y::NoiseGenerator{T}) where {T} = NoiseTuple((x.layers..., y))
+(x::NoiseGenerator{T}, y::NoiseTuple{T}) where {T} = NoiseTuple(x, y.layers...)
+(x::NoiseTuple{T}, y::NoiseTuple{T}) where {T} = NoiseTuple(x.layers, y.layers...)

for fname ∈ (
    :(Base.getindex), 
    :(Base.length), 
    :(Base.first), 
    :(Base.last), 
    :(Base.iterate), 
    :(Base.lastindex),
    :(Base.firstindex),
    )

    @eval $fname(noise::NoiseTuple, args...; kws...) = $fname(noise.layers, args...; kws...)
    
end


isnoise(noise::NoiseGenerator{T}) where {T} = !(typeof(noise) == NoNoise{T})
