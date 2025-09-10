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
# Contains types defining affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export AffineCouplingElement, AffineCouplingLayer
export AffineCouplingAxes
export AffineCouplingBlock, AffineCouplingChain
export AffineCouplingTest

struct AffineCouplingAxes
    
    d::Int # total number of dimensions without the conditions
    n::Int # number of conditions
    
    axis_id::Vector{Int}
    axis_af::Vector{Int}
    axis_nn::Vector{Int}

    reverse::Bool
    
end


function Base.show(io::IO, obj::AffineCouplingAxes)
    print(io, "d = $(obj.d), n (params) = $(obj.n), unmodified dims = $(obj.axis_id), modified dims = $(obj.axis_af)")
end



@doc raw"""

    AffineCouplingElement <: FlowElement

"""
abstract type AffineCouplingElement <: FlowElement end 
abstract type AffineCouplingLayer <: AffineCouplingElement end

############
# Affine block layer structure

struct AffineCouplingBlock{T<:AffineCouplingLayer, U<:AffineCouplingLayer} <: AffineCouplingElement
    layer_1::T
    layer_2::U
end

Flux.@layer AffineCouplingBlock
Functors.@functor AffineCouplingBlock

Base.length(obj::AffineCouplingBlock) = 2

function Base.show(io::IO, obj::AffineCouplingBlock, n::Int = 1)
    show(io, obj.layer_1, n)
    show(io, obj.layer_2, n+1)
end


############
# Affine chain structure

struct AffineCouplingChain{T<:Union{Tuple, AbstractVector}} <: AffineCouplingElement
    layers::T
end

Flux.@layer AffineCouplingChain
Functors.@functor AffineCouplingChain

Base.length(obj::AffineCouplingChain) = length(obj.layers)

function Base.show(io::IO, obj::AffineCouplingChain, n::Int = 1)
    l = 0
    for layer in obj.layers
        show(io, layer, n+l)
        l = l + length(layer)
    end
end


###########
# Test structure

struct AffineCouplingTest{T<:Union{Tuple, AbstractVector}, U<:AffineCouplingElement} <: AffineCouplingElement
    layers::T
    single::U
end