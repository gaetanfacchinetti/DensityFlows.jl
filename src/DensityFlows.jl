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
import Functors
import Distributions
import Optimisers
import Random
import LinearAlgebra
import JLD2
import ChainRulesCore

import Base: reverse
import ChainRulesCore: rrule
import Distributions: sample, logpdf, pdf
import Flux: Dense

export FlowElement, CouplingLayer, summarize

@doc raw""" Building blocks of the flow """ 
abstract type FlowElement end
abstract type CouplingLayer <: FlowElement end

# default behaviour for summarize
@doc raw"""

    summarize(element)

Print a summary for the [`FlowElement`](@ref) `element`.
"""
summarize(element::FlowElement) = println(element)


include("./Macros.jl")
include("./Data.jl")
include("./Axes.jl")
include("./affine/RNVP.jl")
include("./affine/NICE.jl")
include("./norm/Normalization.jl")
include("./Layers.jl")
include("./Blocks.jl")
include("./Chains.jl")
include("./Flows.jl")
include("./Loading.jl")

@flow_wrapper backward forward forward!
@unconditional_wrapper backward forward forward!
@unconditional_wrapper predict pdf logpdf

end
