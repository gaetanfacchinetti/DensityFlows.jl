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
# Contains functions related to affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export MetaData, DataPartition, DataArrays


struct MetaData{T<:AbstractVector}
    hash::String

    x_min::T
    x_max::T

    θ_min::T
    θ_max::T
end

struct DataPartition{T<:AbstractVector{Int}}
    training::T
    validation::T
    testing::T
end

function DataPartition(
    n::Int, 
    f_training::Real = 0.9, 
    f_validation::Real = 0.1, 
    rng::Random.AbstractRNG = Random.default_rng()
    )

    # generate a random permutation of length n
    p = Random.randperm(rng, n)

    # define chunk edges to partition the permutation
    # these chunks correspond to training, validation and testing
    i1 = round(Int, n * f_training)
    i2 = i1 + round(Int, n * f_validation)

    return DataPartition(p[1:i1], p[i1+1:i2], p[i2+1:n])
end


struct DataArrays{T, N, A<:AbstractArray{T, N}, B<:AbstractVector{<:Int}, C<:AbstractVector{T}}

    x::A # raw data
    y::A # normalized data
    
    θ::A # raw parameters
    t::A # normalised parameters
      
    partition::DataPartition{B}
    metadata::MetaData{C}

end


function DataArrays(
    x::AbstractArray{T, N}, 
    θ::Union{AbstractArray{T, N}, Nothing} = nothing,
    f_training::Real = 0.8, 
    f_validation::Real = 0.1, 
    rng::Random.AbstractRNG = Random.default_rng()
    ) where {T, N}
    
    x_min = vec(minimum(x, dims=2:N))
    x_max = vec(maximum(x, dims=2:N))

    y = normalize_input(x, x_min, x_max)

    if θ !== nothing
        
        θ_min = vec(minimum(θ, dims=2:N))
        θ_max = vec(maximum(θ, dims=2:N))

        t = normalize_input(θ, θ_min, θ_max)
    else

        θ = Array{T, N}(undef, 0, size(x)[2:end]...)
        t = Array{T, N}(undef, 0, size(x)[2:end]...)

        θ_min = vec(minimum(θ, dims=2:N))
        θ_max = vec(maximum(θ, dims=2:N))
    end

    partition = DataPartition(size(y, 2), f_training, f_validation, rng)
    metadata = MetaData("", x_min, x_max, θ_min, θ_max)

    return DataArrays(x, y, θ, t, partition, metadata)

end


training_data(data::DataArrays{T, N}) where {T, N} = data.y[:, data.partition.training, ntuple(_ -> :, N-2)...], data.t[:, data.partition.training, ntuple(_ -> :, N-2)...]
validation_data(data::DataArrays{T, N}) where {T, N} = data.y[:, data.partition.validation, ntuple(_ -> :, N-2)...], data.t[:, data.partition.validation, ntuple(_ -> :, N-2)...]
testing_data(data::DataArrays{T, N}) where {T, N} = data.y[:, data.partition.testing, ntuple(_ -> :, N-2)...], data.t[:, data.partition.testing, ntuple(_ -> :, N-2)...]

normalize_input(x::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T} = (x .- x_min) ./ (x_max .- x_min)
resize_output(y::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T} = (x_max .- x_min) .* y .+ x_min


normalize_input(x::Nothing, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T} = nothing
grad_normalisation(x_min::T, x_max::T) where {T} = 1 ./ (x_max - x_min)


@inline function normalize_input!(x::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T}
    x .= (x .- x_min) ./ (x_max .- x_min)
end

@inline function resize_output!(y::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T}
    y .= (x_max .- x_min) .* y .+ x_min
end