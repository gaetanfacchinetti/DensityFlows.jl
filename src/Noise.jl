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

export NoiseModel, NoiseGenerator, NoNoise, GaussianNoise, TruncatedGaussianNoise, ResampledNoise
export noisy_data

abstract type NoiseModel{T} end
abstract type NoiseGenerator{T} <: NoiseModel{T} end

struct NoNoise{T} <: NoiseGenerator{T} end

NoNoise(::Type{T}) where {T} = NoNoise{T}()
NoNoise() = NoNoise{Float32}()

(f::NoNoise)(x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = (x, θ)

struct GaussianNoise{T, V<:AbstractVector{T}} <: NoiseGenerator{T}
    μx::V
    σx::V
    μθ::V
    σθ::V
end

noisy_data(noise::NoiseGenerator{T}, x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = noise(x, θ)
noisy_data(x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = (x, θ)

## GAUSSIAN NOISE

# by default define a noise in the parameters
function _GaussianNoise(
    μx::Union{T, AbstractVector{T}}, 
    σx::Union{T, AbstractVector{T}}, 
    μθ::Union{T, AbstractVector{T}}, 
    σθ::Union{T, AbstractVector{T}}
    ) where {T<:AbstractFloat}

    μxp = (μx isa AbstractVector) ? μx : [μx]
    σxp = (σx isa AbstractVector) ? σx : [σx]
    μθp = (μθ isa AbstractVector) ? μθ : [μθ]
    σθp = (σθ isa AbstractVector) ? σθ : [σθ]

    GaussianNoise(μxp, σxp, μθp, σθp)
end

GaussianNoise() = _GaussianNoise(0f0, 0f0, 0f0, 0f0)
GaussianNoise(σ::Union{T, AbstractVector{T}}) where {T<:AbstractFloat} = _GaussianNoise(zeros(T, 3)..., σ)
GaussianNoise(μ::Union{T, AbstractVector{T}}, σ::Union{T, AbstractVector{T}}) where {T<:AbstractFloat} = _GaussianNoise(zeros(T, 2)..., μ, σ)

function (f::GaussianNoise{T})(x::AbstractArray{T}, θ::AbstractArray{T}) where {T}
    return (x .+ muladd.(f.σx, randn(T, size(x)...), f.μx),  (θ .+ muladd.(f.σθ, randn(T, size(θ)...), f.μθ)))
end

## TRUNCATED GAUSSIAN NOISE

struct TruncatedGaussianNoise{T, V<:AbstractVector{T}} <: NoiseGenerator{T}
    
    μx::V
    σx::V
    μθ::V
    σθ::V

    x_min::V
    x_max::V
    θ_min::V
    θ_max::V

end

# by default define a noise in the parameters
function _TruncatedGaussianNoise(
    μx::Union{T, AbstractVector{T}}, 
    σx::Union{T, AbstractVector{T}}, 
    μθ::Union{T, AbstractVector{T}}, 
    σθ::Union{T, AbstractVector{T}},
    x_min::Union{T, AbstractVector{T}}, 
    x_max::Union{T, AbstractVector{T}}, 
    θ_min::Union{T, AbstractVector{T}}, 
    θ_max::Union{T, AbstractVector{T}}
    ) where {T<:AbstractFloat}

    μxp = (μx isa AbstractVector) ? μx : [μx]
    σxp = (σx isa AbstractVector) ? σx : [σx]
    μθp = (μθ isa AbstractVector) ? μθ : [μθ]
    σθp = (σθ isa AbstractVector) ? σθ : [σθ]

    x_min_p = (x_min isa AbstractVector) ? x_min : [x_min]
    x_max_p = (x_max isa AbstractVector) ? x_max : [x_max]
    θ_min_p = (θ_min isa AbstractVector) ? θ_min : [θ_min]
    θ_max_p = (θ_max isa AbstractVector) ? θ_max : [θ_max]

    TruncatedGaussianNoise(μxp, σxp, μθp, σθp, x_min_p, x_max_p, θ_min_p, θ_max_p)
end

TruncatedGaussianNoise() = _TruncatedGaussianNoise(0f0, 0f0, 0f0, 0f0, -Inf32, Inf32, -Inf32, Inf32)
TruncatedGaussianNoise(σ::Union{T, AbstractVector{T}}, min::Union{T, AbstractVector{T}}, max::Union{T, AbstractVector{T}}) where {T} = _TruncatedGaussianNoise(zeros(T, 3)..., σ, -T(Inf), T(Inf), min, max)

function (f::TruncatedGaussianNoise{T})(x::AbstractArray{T}, θ::AbstractArray{T}) where {T}
    
    α_θ = (f.θ_min .- (f.μθ .+ θ)) ./ f.σθ
    β_θ = (f.θ_max .- (f.μθ .+ θ)) ./ f.σθ
    Φ_α_θ = (1 .+ SpecialFunctions.erf.(α_θ/sqrt(2)))/2
    Φ_β_θ = (1 .+ SpecialFunctions.erf.(β_θ/sqrt(2)))/2

    tθ = Random.rand(size(θ)...)
    y = Φ_α_θ .* (1 .- tθ) .+ Φ_β_θ .* tθ
    θ_new = (f.μθ .+ θ) .+ f.σθ .* sqrt(2) .* SpecialFunctions.erfinv.(2*y .- 1)


    α_x = (f.x_min .- (f.μx .+ x)) ./ f.σx
    β_x = (f.x_max .- (f.μx .+ x)) ./ f.σx
    Φ_α_x = (1 .+ SpecialFunctions.erf.(α_x/sqrt(2)))/2
    Φ_β_x = (1 .+ SpecialFunctions.erf.(β_x/sqrt(2)))/2

    tx = Random.rand(size(x)...)
    y = Φ_α_x .* (1 .- tx) .+ Φ_β_x .* tx
    x_new = (f.μx .+ x) .+ f.σx .* sqrt(2) .* SpecialFunctions.erfinv.(2*y .- 1)

    return (T.(x_new), T.(θ_new))
end


## RESAMPLE NOISE

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
