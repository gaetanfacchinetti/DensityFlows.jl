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
# Contains functions related to the affine coupling flows
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


############
# AffineFlow chain structure

struct AffineCouplingFlow{M<:AffineCouplingChain, D<:Distributions.Distribution} <: Flow{M, D}
    
    model::M
    base::D
end

function AffineCouplingFlow(
    n_couplings::Int, 
    ::Type{U} = AffineCouplingBlock,
    ::Type{T} = Float32,
    base::Distributions.Distribution = Distributions.MvNormal( zeros(T, axes.d), diagm(ones(T, axes.d)));
    kws...
    ) where {T<:AbstractFloat, U<:AffineCouplingInstance}
    
    return AffineCouplingFlow(AffineCouplingChain(n_couplings, axes, U; kws...), base)

end


backward(flow::AffineCouplingFlow, x::AbstractArray{T}, θ::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat} = backward(flow.model, x, θ)
forward(flow::AffineCouplingFlow, x::AbstractArray{T}, θ::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat} = forward(flow.model, x, θ)
