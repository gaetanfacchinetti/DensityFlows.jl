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
export minimum_θ, maximum_θ
export normalize_input, normalize_input!
export resize_output, resize_output!
export dflt_θ


@doc raw"""

    dflt_θ([T = Float32,] dims::Tuple)
    dflt_θ([T = Float32,] dims...)
    dflt_θ(x::AbstractArray)

Default value of the parameters, setting the first dimension to size 0.

Arguments are the same than `zeros` or `ones`. If an array `x` is passed
returns an array of the same dimensions than `x` with the first dimension
set to size 0.

# Examples
```jldoctest
julia> dflt_θ(2, 3)
0×2×3 Array{Float32, 3}

julia> dflt_θ(ones(2, 4, 5))
0×4×5 Array{Float64, 3}
```
"""
function dflt_θ end

dflt_θ(dims::Base.DimOrInd...) = dflt_θ(dims)
dflt_θ(::Type{T}, dims::Base.DimOrInd...) where {T} = dflt_θ(T, dims)
dflt_θ(dims::Tuple{Vararg{Base.DimOrInd}}) = dflt_θ(Float32, dims)
dflt_θ(::Type{T}, dims::NTuple{N, Union{Integer, Base.OneTo}}) where {T, N} = dflt_θ(Float32, map(Base.to_dim, dims))
dflt_θ(x::AbstractArray{T, N}) where {T,N} = dflt_θ(T, size(x)[2:N])

function dflt_θ(::Type{T}, dims::NTuple{N, Integer}) where {T,N}
    return Array{T,N+1}(undef, (0, dims...))
end



@doc raw"""

    MetaData(hash, θ_min, θ_max)

Metadata containing an identification hash value and the boundaries of the parameters array θ.
"""
struct MetaData{T, V<:AbstractVector{T}}
    
    hash::String

    # same information as in the
    d::Integer
    n::Integer

    θ_min::V
    θ_max::V

end


@doc raw"""Minimin value of the input parameters."""
minimum_θ(metadata::MetaData) = metadata.θ_min

@doc raw"""Maximum value of the input parameters."""
maximum_θ(metadata::MetaData) = metadata.θ_max


struct DataPartition{T<:AbstractVector{Int}}
    training::T
    validation::T
    testing::T
end

@doc raw"""
    
    DataPartition(n, f_training = 0.9, f_validation = 0.1, rng = Random.default_rng())

Random partition of the data.

The data is devided into a fraction `f_training` of training data and 
a fraction `f_validation` of validation data. If `f_training + f_validation < 1`
the rest is kept as testing data. 
"""
function DataPartition(
    n::Integer, 
    f_training::T = 0.9, 
    f_validation::T = 0.1, 
    rng::Random.AbstractRNG = Random.default_rng()
    ) where {T}

    # generate a random permutation of length n
    p = Random.randperm(rng, n)

    # define chunk edges to partition the permutation
    # these chunks correspond to training, validation and testing
    i1 = round(Int, n * f_training)
    i2 = i1 + round(Int, n * f_validation)

    return DataPartition(p[1:i1], p[i1+1:i2], p[i2+1:n])
end


struct DataArrays{T, N, A<:AbstractArray{T, N}, B<:AbstractArray{T, N}, C<:AbstractVector{<:Int}}

    x::A # raw data
    θ::B # raw parameters
      
    partition::DataPartition{C}

end

@doc raw"""

    DataArrays(x, θ = dflt_θ(x); f_training = 0.9, f_validation = 0.1, rng = Random.default_rng())

Normalised and partitioned data to feed the neural network.

`x` must be of size `(d, ...)` where `d` is the number of physical dimensions. 
`θ` must be of size `(n, ...)`where `n` is the number of parameters and every 
other array dimensions in place of `...` should match that of `x`. 

!!! warning
    Data is partitioned along the second axis only. It is thus necessary to make sure that `size(x, 2)` 
    is large enough by swapping some of the dimensions if necessary.

See also [`DataPartition`](@ref) and [`MetaData`](@ref).
"""
function DataArrays(
    x::AbstractArray{T, N}, 
    θ::AbstractArray{T, N} = dflt_θ(x);
    f_training::X = 0.9, 
    f_validation::X = 0.1, 
    rng::Random.AbstractRNG = Random.default_rng()
    ) where {T, N, X}

    @assert length(size(x)) >= 2 "data must be an array of size (d, i1, ...) at least"
    @assert all(size(x)[2:N] .== size(θ)[2:N]) "x and θ must have the same size -- except for the first dimension"
    
    partition = DataPartition(size(x, 2), f_training, f_validation, rng)
    return DataArrays{T, N, typeof(x), typeof(θ), typeof(partition.training)}(x, θ, partition)

end


function summarize(obj::DataArrays)
    println("Data with size $(size(obj.x)) and parameters / conditions with size $(size(obj.θ)).")
    println("-> f_training = $(length(obj.partition.training)/size(obj.x, 2)), f_validation = $(length(obj.partition.validation)/size(obj.x, 2)).")
end


number_dimensions(data::DataArrays) = size(data.x, 1)
number_conditions(data::DataArrays) = size(data.θ, 1)

minimum_θ(data::DataArrays{T, N}) where {T, N} = vec(minimum(data.θ, dims=2:N))
maximum_θ(data::DataArrays{T, N}) where {T, N} = vec(maximum(data.θ, dims=2:N))

training_data(data::DataArrays)   = selectdim(data.x, 2, data.partition.training),   selectdim(data.θ, 2, data.partition.training) 
validation_data(data::DataArrays) = selectdim(data.x, 2, data.partition.validation), selectdim(data.θ, 2, data.partition.validation)
testing_data(data::DataArrays)    = selectdim(data.x, 2, data.partition.testing),    selectdim(data.θ, 2, data.partition.testing)

function normalized_training_data(data::DataArrays, metadata::MetaData) 
    x, θ = training_data(data)
    t = normalize_input(θ, metadata.θ_min, metadata.θ_max)
    return x, t
end

function normalized_validation_data(data::DataArrays, metadata::MetaData) 
    x, θ = validation_data(data)
    t = normalize_input(θ, metadata.θ_min, metadata.θ_max)
    return x, t
end

@doc raw"""

    normalise_input(x, x_min, x_max)

Normalize input between ``[-1, 1]``. 

```math
    y = \frac{x - x_{\rm min}}{x_{\rm max} - x_{\rm min}}
```

See also [`normalize_input!`](@ref) and [`resize_output`](@ref).
"""
function normalize_input(x::AbstractArray{T, N}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T, N} 
    x_diff = x_max .- x_min
    y = (x .- x_min) ./ x_diff
    @views y[x_diff .== zero(T), ntuple(_ -> :, N-1)...] .= zero(T)
    return y
end

@doc raw"""

    normalise_input(x, x_min, x_max)

Resize the output between ``[x_{\rm min}, x_{\rm max}]``. 

```math
    x = (x_{\rm max} - x_{\rm min}) y + x_{\rm min}
```

See also [`resize_output!`](@ref) and [`normalize_input`](@ref).
"""
resize_output(y::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T} = (x_max .- x_min) .* y .+ x_min



function grad_normalisation(x_min::T, x_max::T) where {T} 
    x_diff = x_max .- x_min
    res = 1 ./ x_diff
    @views res[x_diff .== zero(T), ntuple(_ -> :, N-1)...] .= zero(T)
    return res
end


@doc raw"""

    normalise_input!(x, x_min, x_max)

Normalize input between ``[-1, 1]`` in place. 

See also [`normalize_input`](@ref) and [`resize_output!`](@ref).
"""
@inline function normalize_input!(x::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T}
    x_diff = x_max .- x_min
    x .= (x .- x_min) ./ x_diff
    @views x[x_diff .== zero(T), ntuple(_ -> :, N-1)...] .= zero(T)
end


@doc raw"""

    normalise_input!(x, x_min, x_max)

Resize the output between ``[x_{\rm min}, x_{\rm max}]`` in place. 

See also [`resize_output`](@ref) and [`normalize_input!`](@ref).
"""
@inline function resize_output!(y::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T}
    y .= (x_max .- x_min) .* y .+ x_min
end