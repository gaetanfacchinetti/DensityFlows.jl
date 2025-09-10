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


struct NormalizationLayer{T<:AbstractFloat, U<:AbstractArray{T}} <: FlowElement
    x_min::U
    x_max::U
end

function NormalizationLayer(x::AbstractArray{T, N}) where {T, N}
    x_min = vec(minimum(x, dims=2:N))
    x_max = vec(maximum(x, dims=2:N))
    return NormalizationLayer(x_min, x_max)
end

Flux.@layer NormalizationLayer
Functors.@functor NormalizationLayer

Optimisers.trainable(m::NormalizationLayer) = (;)

function DensityFlows.backward(
    nlayer::NormalizationLayer, 
    x::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    z = (x .- nlayer.x_min) ./ (nlayer.x_max .- nlayer.x_min)
    ln_det_jac = - sum(log.((nlayer.x_max .- nlayer.x_min)))

    return z, ln_det_jac
end

function DensityFlows.forward(
    nlayer::NormalizationLayer, 
    z::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    x = (nlayer.x_max .- nlayer.x_min) .* z  .+ nlayer.x_min 
    ln_det_jac = + sum(log.((nlayer.x_max .- nlayer.x_min)))

    return x, ln_det_jac
end


function DensityFlows.forward!(
    nlayer::NormalizationLayer, 
    z::AbstractArray{T, N},
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    z = (nlayer.x_max .- nlayer.x_min) .* z  .+ nlayer.x_min 

end

function DensityFlows.save(filename::String, array::AbstractArray)
    
    try
        JLD2.jldsave(filename * ".jld2"; array)
    catch e
        println("Impossible to save the normalization")
        rethrow(e)
    end

end

function load(filename::String, ::Type{AbstractArray})

    try
        data = JLD2.jldopen(filename * ".jld2")
        return data["array"]
    catch e
        println("Impossible to load NormalizationLayer at $filename")
        rethrow(e)
    end

end


function Base.show(io::IO, obj::NormalizationLayer, n::Int = 1)
    println(io, "• layer_$n -> normalization layer")
end
