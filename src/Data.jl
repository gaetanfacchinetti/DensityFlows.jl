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
export data_x_min, data_x_max, data_θ_min, data_θ_max
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

    MetaData(hash, x_min, x_max, θ_min, θ_max)

Metadata containing an identification hash value and the boundaries of the
data and parameters arrays x and θ.
"""
struct MetaData{T<:AbstractVector}
    hash::String

    x_min::T
    x_max::T

    θ_min::T
    θ_max::T
end

@doc raw"""Minimum value of the input data."""
data_x_min(metadata::MetaData) = metadata.x_min

@doc raw"""Maximum value of the input data."""
data_x_max(metadata::MetaData) = metadata.x_max

@doc raw"""Minimin value of the input parameters."""
data_θ_min(metadata::MetaData) = metadata.θ_min

@doc raw"""Maximum value of the input parameters."""
data_θ_max(metadata::MetaData) = metadata.θ_max


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

@doc raw"""

    DataArrays(x, θ = nothing, f_training = 0.9, f_validation = 0.1, rng = Random.default_rng())

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
    θ::Union{AbstractArray{T, N}, Nothing} = nothing,
    f_training::Real = 0.9, 
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

function Base.show(io::IO, obj::DataArrays)
    println(io, "Data with size $(size(obj.x)) and parameters / conditions with size $(size(obj.θ)).")
    println(io, "-> f_training = $(length(obj.partition.training)/size(obj.x, 2)), f_validation = $(length(obj.partition.validation)/size(obj.x, 2)).")
end


training_data(data::DataArrays{T, N}) where {T, N} = data.y[:, data.partition.training, ntuple(_ -> :, N-2)...], data.t[:, data.partition.training, ntuple(_ -> :, N-2)...]
validation_data(data::DataArrays{T, N}) where {T, N} = data.y[:, data.partition.validation, ntuple(_ -> :, N-2)...], data.t[:, data.partition.validation, ntuple(_ -> :, N-2)...]
testing_data(data::DataArrays{T, N}) where {T, N} = data.y[:, data.partition.testing, ntuple(_ -> :, N-2)...], data.t[:, data.partition.testing, ntuple(_ -> :, N-2)...]

@doc raw"""

    normalise_input(x, x_min, x_max)

Normalize input between ``[-1, 1]``. 

```math
    y = \frac{x - x_{\rm min}}{x_{\rm max} - x_{\rm min}}
```

See also [`normalize_input!`](@ref) and [`resize_output`](@ref).
"""
function normalize_input(x::AbstractArray{T, N}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T, N} 
    y = (x .- x_min) ./ (x_max .- x_min)
    y[vec((x_max .- x_min) .== zero(T)), ntuple(_ -> :, N-1)...] .= zero(T)
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

normalize_input(x::Nothing, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T} = nothing
grad_normalisation(x_min::T, x_max::T) where {T} = 1 ./ (x_max - x_min)

@doc raw"""

    normalise_input!(x, x_min, x_max)

Normalize input between ``[-1, 1]`` in place. 

See also [`normalize_input`](@ref) and [`resize_output!`](@ref).
"""
@inline function normalize_input!(x::AbstractArray{T, N}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T, N}
    x .= (x .- x_min) ./ (x_max .- x_min)
    x[vec((x_max .- x_min) .== zero(T)), ntuple(_ -> :, N-1)...] .= zero(T)
end


@doc raw"""

    normalise_input!(x, x_min, x_max)

Resize the output between ``[x_{\rm min}, x_{\rm max}]`` in place. 

See also [`resize_output`](@ref) and [`normalize_input!`](@ref).
"""
@inline function resize_output!(y::AbstractArray{T}, x_min::AbstractVector{T}, x_max::AbstractVector{T}) where {T}
    y .= (x_max .- x_min) .* y .+ x_min
end