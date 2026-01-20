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
# Contains functions related to probability distributions
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export Distribution, GaussianDistribution, StandardGaussianDistribution, rand

abstract type Distribution{T,D} end

# DIAGONAL GAUSSIAN DISTRIBUTION

struct GaussianDistribution{T,D,V<:AbstractVector{T}} <: Distribution{T,D}
    μ::V
    σ::V
end

function GaussianDistribution(μ::V, σ::V) where {T, V<:AbstractVector{T}} 
    @assert length(μ) == length(σ)
    return GaussianDistribution{T, length(σ), typeof(μ)}(μ, σ)
end

GaussianDistribution(σ::AbstractVector{T}) where {T} = GaussianDistribution(zeros(T, length(σ)), Vector{T}(σ))
GaussianDistribution(::Type{T}, d::Int) where {T} = GaussianDistribution(zeros(T, d), ones(T, d))
GaussianDistribution(d::Int) = GaussianDistribution(zeros(Float32, d), ones(Float32, d))

function logpdf(dist::GaussianDistribution{T, D}, z::AbstractArray{T}) where {T,D} 
    z2_red = dropdims(sum(((z .- dist.μ) / dist.σ).^2, dims=1), dims=1)
    logσ_red = dropdims(sum(log.(dist.σ), dims=1), dims=1)
    return -( z2_red .+ D * log.(T(2π))  .+ logσ_red) / 2
end

function rand(rng::Random.AbstractRNG, dist::GaussianDistribution{T,D}, dims::Tuple{Vararg{Integer}}) where {T,D}
    return dist.μ .+ dist.σ .* Random.randn(rng, T, (D, dims...))
end


# STANDARD GAUSSIAN DISTRIBUTION

struct StandardGaussianDistribution{T,D} <: Distribution{T,D} end

StandardGaussianDistribution(::Type{T}, d::Int) where {T} = StandardGaussianDistribution{T,d}()
StandardGaussianDistribution(d::Int) = StandardGaussianDistribution(Float32, d)

function logpdf(dist::StandardGaussianDistribution{T,D}, z::AbstractArray{T}) where {T,D} 
    @assert size(z, 1) == D
    z2_red = dropdims(sum(z.^2, dims=1), dims=1)
    return - (z2_red  .+ D * log(T(2π))) / 2
end

function rand(rng::Random.AbstractRNG, dist::StandardGaussianDistribution{T,D}, dims::Tuple{Vararg{Integer}}) where {T,D}
    return Random.randn(rng, T, (Int(D), dims...))
end



# RANDOM DRAWS

rand(rng::Random.AbstractRNG, dist::Distribution, dims::Integer)  = rand(rng, dist, (dims,))
rand(rng::Random.AbstractRNG, dist::Distribution, dims::Integer...)  = rand(rng, dist, dims)
rand(dist::Distribution, dims::Tuple{Vararg{Integer}}) = rand(Random.default_rng(), dist, dims)
rand(dist::Distribution, dims::Integer...) = rand(Random.default_rng(), dist, dims)
rand(dist::Distribution, dims::Integer) = rand(Random.default_rng(), dist, (dims,))
