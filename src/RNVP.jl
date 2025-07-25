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
# Contains functions related to real-NVP affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################




function backward(
    layer::RNVPCouplingLayer, 
    x::AbstractArray{T, N}, 
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    inds_nn = (layer.axes.axis_nn, ntuple(_ -> :, N-1)...)
    inds_id = (layer.axes.axis_id, ntuple(_ -> :, N-1)...)
    inds_af = (layer.axes.axis_af, ntuple(_ -> :, N-1)...)
    
    if θ === nothing || size(θ, 1) == 0
        @views input = x[inds_nn...]
    else
        @views input = vcat(θ, x)[inds_nn...]
    end
    
    s = layer.s_net(input)
    t = layer.t_net(input)

    # Compute log-det-Jacobian
    ln_det_jac = - dropdims(sum(s, dims = 1), dims = 1)

    # check if we are using a reverse layer of not
    # in a reverse layer the modyfied entries are
    # before the one left unchanged

    if !(layer.axes.reverse)
        # if not reverse the identity comes before the affine transformation
        return (@views vcat(x[inds_id...], (x[inds_af...] .- t) .* exp.(-s))), ln_det_jac
    end

    # otherwise it is the opposite
    return (@views vcat((x[inds_af...] .- t) .* exp.(-s), x[inds_id...])), ln_det_jac
    
end



function forward(
    layer::RNVPCouplingLayer, 
    z::AbstractArray{T, N}, 
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    inds_nn = (layer.axes.axis_nn, ntuple(_ -> :, N-1)...)
    inds_id = (layer.axes.axis_id, ntuple(_ -> :, N-1)...)
    inds_af = (layer.axes.axis_af, ntuple(_ -> :, N-1)...)
    
    if θ === nothing || size(θ, 1) == 0
        @views input = z[inds_nn...]
    else
        @views input = vcat(θ, z)[inds_nn...]
    end

    s = layer.s_net(input) # output of size (d-j, n_samples) or (j, n_samples)
    t = layer.t_net(input) # output of size (d-j, n_samples) or (j, n_samples)

    # ln|det J[T^{-1}]|
    ln_det_jac = dropdims(sum(s, dims = 1), dims = 1)

    # check if we are using a reverse layer of not
    # in a reverse layer the modyfied entries are
    # before the one left unchanged

    if !(layer.axes.reverse)
        return (@views vcat((z[inds_id...], z[inds_af...] .* exp.(s) .+ t )...)), ln_det_jac
    end
        
    return (@views vcat((z[inds_af...] .* exp.(s) .+ t, z[inds_id...])...)), ln_det_jac

end


function forward!(
    layer::RNVPCouplingLayer, 
    z::AbstractArray{T, N}, 
    θ::Union{AbstractArray{T, N}, Nothing} = nothing
    ) where {T<:AbstractFloat, N}

    inds_nn = (layer.axes.axis_nn, ntuple(_ -> :, N-1)...)
    inds_af = (layer.axes.axis_af, ntuple(_ -> :, N-1)...)
    
    if θ === nothing || size(θ, 1) == 0
        @views input = z[inds_nn...]
    else
        @views input = vcat(θ, z)[inds_nn...]
    end

    s = layer.s_net(input) # output of size (d-j, n_samples) or (j, n_samples)
    t = layer.t_net(input) # output of size (d-j, n_samples) or (j, n_samples)

    z[inds_af...] .= z[inds_af...] .* exp.(s) .+ t 

end


function (f::RNVPCouplingLayer)(z::AbstractArray{T, N}, θ::Union{AbstractArray{T, N}, Nothing} = nothing) where {T<:AbstractFloat, N}
    return forward(f, z, θ)
end

