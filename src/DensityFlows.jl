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
# Contains functions related to the density normalizing flows
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


module DensityFlows

import Flux
import Optimisers
import Random
import LinearAlgebra
import JLD2
import ChainRulesCore
import Statistics

import Base: reverse, ==, +
import ChainRulesCore: rrule
import Flux: Dense

export FlowElement, CouplingLayer, summarize

@doc raw""" Building blocks of the flow """ 
abstract type FlowElement{T,D,N} end

# make the difference between containers of multiple simple flows
abstract type CouplingLayer{T,D,N} <: FlowElement{T,D,N} end

# default behaviour for summarize
@doc raw"""

    summarize(element)

Print a summary for the [`FlowElement`](@ref) `element`.
"""
summarize(element::FlowElement) = println(element)


number_dimensions(element::FlowElement{T,D,N}) where {T,D,N} = D
number_conditions(element::FlowElement{T,D,N}) where {T,D,N} = N

include("./FluxWrapper.jl")
include("./Macros.jl")
include("./Distributions.jl")
include("./Data.jl")
include("./condition/Conditioners.jl")
include("./Axes.jl")
include("./affine/RNVP.jl")
include("./affine/NICE.jl")
include("./affine/CMAF.jl")
include("./norm/GlobalNormalization.jl")
include("./norm/BatchNormalization.jl")
#include("./Blocks.jl")
include("./Chains.jl")
include("./Flows.jl")
include("./Noise.jl")
include("./Training.jl")
include("./Loading.jl")

@flow_wrapper backward forward forward!
@unconditional_wrapper backward forward forward!

end
