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

export AffineCouplingFlow, predict, train!

struct AffineCouplingFlow{M<:AffineCouplingChain, D<:Distributions.Distribution, T<:AbstractFloat, U<:AbstractArray{T}} <: Flow{M, D}
    
    model::M
    base::D

    metadata::MetaData{U}

    train_loss::Vector{T}
    valid_loss::Vector{T}

end


function AffineCouplingFlow(
    n_couplings::Int,
    metadata::MetaData{<:AbstractArray{T}}, 
    ::Type{U} = AffineCouplingBlock,
    base::Union{Distributions.Distribution, Nothing} = nothing;
    kws...
    ) where {T<:AbstractFloat, U<:AffineCouplingInstance}
    
    # the dimension is given by the size of x_min
    d = size(metadata.x_min, 1)
    n = size(metadata.θ_min, 1)

    axes = AffineCouplingAxes(d, n)
    chain = AffineCouplingChain(n_couplings, axes, U; kws...)

    if base === nothing
        base = Distributions.MvNormal(zeros(T, d), LinearAlgebra.diagm(ones(T, d)))
    end

    return AffineCouplingFlow(chain, base, metadata, T[], T[])

end


backward(flow::AffineCouplingFlow, y::AbstractArray{T}, t::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat} = backward(flow.model, y, t)
forward(flow::AffineCouplingFlow, z::AbstractArray{T}, t::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat} = forward(flow.model, z, t)

# work with the non renormalized quantities here
function predict(flow::AffineCouplingFlow, z::AbstractArray{T}, θ::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat}
    
    if θ === nothing
        return resize_output(forward(flow, z, nothing)[1], flow.metadata.x_min, flow.metadata.x_max)
    end

    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max)
    return resize_output(forward(flow, z, t)[1], flow.metadata.x_min, flow.metadata.x_max)
end


function sample(
    flow::AffineCouplingFlow,
    n::Int = 1, 
    θ::Union{AbstractArray{T}, Nothing} = nothing,
    rng::Random.AbstractRNG = Random.default_rng()
    ) where {T<:AbstractFloat}
    
    return predict(flow, rand(rng, flow.base, n), θ)
end


function loss(
    z::AbstractArray{T},
    ln_det_jac::AbstractArray{T},
    base::Distributions.Distribution,
    ) where {T<:AbstractFloat}

    return - Statistics.mean(Distributions.logpdf(base, z) .+ ln_det_jac)
end


function train!(
    flow::AffineCouplingFlow,
    data::DataArrays, 
    optimiser_state::NamedTuple; 
    epochs::Int=100,
    batchsize=64,
    shuffle=true,
    verbose::Bool=true
)
    train_data = training_data(data)
    valid_data = validation_data(data)
    
    train_loader = Flux.DataLoader(train_data; batchsize=batchsize, shuffle=shuffle)

    for _ in 1:epochs
        
        for (y_batch, t_batch) in train_loader

            grads = Flux.gradient(flow.model) do m
            
                z, ln_det_jac = backward(m, y_batch, t_batch)
                return loss(z, ln_det_jac, flow.base)
                                
            end

            Optimisers.update!(optimiser_state, flow.model, grads[1])
        end
    
        z, ln_det_jac = backward(flow.model, train_data...)
        train_loss = loss(z, ln_det_jac, flow.base)
        push!(flow.train_loss, train_loss)

        z, ln_det_jac = backward(flow.model, valid_data...)
        valid_loss = loss(z, ln_det_jac, flow.base)
        push!(flow.valid_loss, valid_loss)

        verbose && println("epoch: $(length(flow.train_loss)) | train_loss = $train_loss, valid_loss = $valid_loss")
        
    end

end