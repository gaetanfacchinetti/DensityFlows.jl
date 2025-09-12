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

export AffineCouplingElement
export AffineCouplingLayer
export AffineCouplingAxes
export AffineCouplingBlock

struct AffineCouplingAxes
    
    d::Int # total number of dimensions without the conditions
    n::Int # number of conditions
    
    axis_id::Vector{Int}
    axis_af::Vector{Int}
    axis_nn::Vector{Int}
    
end


function Base.show(io::IO, obj::AffineCouplingAxes)
    print(io, "(d,n)=($(obj.d),$(obj.n)), id=$(obj.axis_id), af=$(obj.axis_af)")
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

    function AffineCouplingBlock(layer_1, layer_2)
        
        # check that the two input layers are compatible
        if !(is_reverse(layer_1.axes, layer_2.axes))
            throw(ArgumentError("When constructing a Block, layers should have opposite / complementary axes"))
        end

        return new{typeof(layer_1), typeof(layer_2)}(layer_1, layer_2)
    end 

end

@auto_flow AffineCouplingBlock
@auto_functor AffineCouplingBlock

Base.length(obj::AffineCouplingBlock) = 2

function Base.show(io::IO, obj::AffineCouplingBlock)
    show(io, obj.layer_1)
    show(io, obj.layer_2)
end

