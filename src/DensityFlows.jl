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

import ChainRulesCore: rrule
import Distributions: sample, logpdf, pdf
import Flux: Dense

export FlowElement

@doc raw""" Building blocks of the flow """ 
abstract type FlowElement end

dflt_θ(::Type{T}, n::Int) where {T} = zeros(T, (0, n))
dflt_θ(::Type{T}, dims::Tuple{Vararg{T}}) where {T} = zeros(T, (0, dims...))
dflt_θ(x::AbstractArray{T, N}) where {T, N} = zeros(T, (0, size(x)[2:N]...))


include("./Macros.jl")
include("./Data.jl")
include("./affine/AffineStructure.jl")
include("./affine/RNVP.jl")
include("./affine/NICE.jl")
include("./affine/AffineCoupling.jl")
include("./Chains.jl")
include("./Flows.jl")
include("./Loading.jl")

end
