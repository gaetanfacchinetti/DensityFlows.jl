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

export Flow, predict, predict!, sample, train!
export logpdf, pdf


##########
#Flow

@doc raw""" Normalizing flow """
struct Flow{T<:AbstractFloat, M<:FlowChain, D<:Distributions.Distribution, U<:AbstractArray{T}}
    
    model::M
    base::D

    metadata::MetaData{U}

    train_loss::Vector{T}
    valid_loss::Vector{T}

end

get_type(flow::Flow{T}) where {T} = T


function Base.show(io::IO, obj::Flow)
    println(io, "- model --------------------")
    show(io, obj.model)
    println(io, "- base distribution --------")
    println(io, "• " * string(Base.typename(typeof(obj.base)).wrapper))
end


function Flow(
    model::FlowChain, 
    base::Distributions.Distribution, 
    data::DataArrays{T}, 
    train_loss::Vector{T}, 
    valid_loss::Vector{T}
    ) where {T}
   
    return  Flow(model, base, data.metadata, train_loss, valid_loss)

end


Flow(model::FlowChain, base::Distributions.Distribution, data::DataArrays{T}) where {T} = Flow(model, base, data.metadata, T[], T[]) 


function Flow(
    n_couplings::Int,
    data::DataArrays{T}, 
    ::Type{U} = AffineCouplingBlock,
    base::Union{Distributions.Distribution, Nothing} = nothing;
    kws...
    ) where {T<:AbstractFloat, U<:AffineCouplingElement}
    
    # the dimension is given by the size of x_min
    d = size(data.metadata.x_min, 1)
    n = size(data.metadata.θ_min, 1)

    # by default build an AffineCouplingFlow
    axes = AffineCouplingAxes(d, n)
    chain = FlowChain(n_couplings, axes, U; kws...)

    if base === nothing
        base = Distributions.MvNormal(zeros(T, d), LinearAlgebra.diagm(ones(T, d)))
    end

    return Flow(chain, base, data.metadata, T[], T[])

end



backward(flow::Flow, y::AbstractArray{T}, t::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat} = backward(flow.model, y, t)
forward(flow::Flow, z::AbstractArray{T}, t::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat} = forward(flow.model, z, t)

@inline function forward!(flow::Flow, z::AbstractArray{T}, t::Union{AbstractArray{T}, Nothing} = nothing) where {T<:AbstractFloat}
    forward!(flow.model, z, t)
end



# work with the non renormalised quantities here
function predict(
    flow::Flow, 
    z::AbstractArray{T}, 
    θ::AbstractArray{T} = dflt_θ(z)
    ) where {T<:AbstractFloat}
    
    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max)
    return resize_output(forward(flow, z, t)[1], flow.metadata.x_min, flow.metadata.x_max)
end


# work with the non renormalised quantities here
function predict!(
    flow::Flow, 
    z::AbstractArray{T}, 
    θ::AbstractArray{T} = dflt_θ(z)
    ) where {T<:AbstractFloat}
    
    if θ === nothing
        forward!(flow, z, nothing)
        resize_output!(z, flow.metadata.x_min, flow.metadata.x_max)
        return
    end

    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max) 

    # compute the forward pass
    forward!(flow, z, t) 

    # resize the output to the non normalised scales
    resize_output!(z, flow.metadata.x_min, flow.metadata.x_max)

end



function sample(
    flow::Flow,
    n::Int = 1,
    θ::AbstractArray{T} = dflt_θ(get_type(flow), n),
    rng::Random.AbstractRNG = Random.default_rng()
    ) where {T<:AbstractFloat}

    r = rand(rng, flow.base, n)
    predict!(flow, r, θ)
    
    return r

end

@doc raw"""
    
    logpdf(flow, x, θ = nothing)

Natural logarithm of the probability density function given by the flow.

See also [`pdf`](@ref).
"""
function logpdf(
    flow::Flow,
    x::AbstractArray{T},
    θ::AbstractArray{T} = dflt_θ(x)
    ) where {T<:AbstractFloat}

    y = normalize_input(x, flow.metadata.x_min, flow.metadata.x_max)

    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max)
    z, ln_det_jac = backward(flow, y, t)
    
    ln_grad_norm = log.(grad_normalisation(flow.metadata.x_min, flow.metadata.x_max))

    return Distributions.logpdf(flow.base, z)  .+ ln_det_jac  .+  sum(ln_grad_norm)

end

@doc raw"""
    
    pdf(flow, x, θ = nothing)

Probability density function given by the flow.

See also [`logpdf`](@ref).
"""
function pdf(
    flow::Flow,
    x::AbstractArray{T},
    θ::AbstractArray{T} = dflt_θ(x)
    ) where {T<:AbstractFloat}

    return exp(logpdf(flow, x, θ))

end



@inline function loss(
    z::AbstractArray{T},
    ln_det_jac::AbstractArray{T},
    base::Distributions.Distribution,
    ) where {T<:AbstractFloat}

    return - Distributions.mean(Distributions.logpdf(base, z) .+ ln_det_jac)
end


function train!(
    flow::Flow,
    data::DataArrays, 
    optimiser_state::NamedTuple; 
    epochs::Int=100,
    batchsize=64,
    shuffle=true,
    verbose::Bool=true,
    debug::Bool=false
)
    train_data = training_data(data)
    valid_data = validation_data(data)
    
    train_loader = Flux.DataLoader(train_data; batchsize=batchsize, shuffle=shuffle)

    for _ in 1:epochs
        
        for (y_batch, t_batch) in train_loader

            grads = Flux.gradient(flow.model) do m
            
                z, ln_det_jac = backward(m, y_batch, t_batch)
                l = loss(z, ln_det_jac, flow.base)

                # debugging if loss gets NaN
                if debug && ( (l != l) || isinf(l))
                    println("$l, $ln_det_jac, $z")
                    throw(ArgumentError(""))
                end

                return l 
                                
            end

            Optimisers.update!(optimiser_state, flow.model, grads[1])

        end
    
        z, ln_det_jac = backward(flow.model, train_data...)
        train_loss = loss(z, ln_det_jac, flow.base)
        push!(flow.train_loss, train_loss)

        if debug && ((train_loss != train_loss) || isinf(train_loss))
            println("Problem with train loss $train_loss")
            return z, ln_det_jac
        end

        z, ln_det_jac = backward(flow.model, valid_data...)
        valid_loss = loss(z, ln_det_jac, flow.base)
        push!(flow.valid_loss, valid_loss)

        if debug && ((valid_loss != valid_loss) || isinf(valid_loss))
            println("Problem with valid loss $valid_loss")
            return z, ln_det_jac
        end

        verbose && println("epoch: $(length(flow.train_loss)) | train_loss = $train_loss, valid_loss = $valid_loss")
        
    end

    if debug
        return nothing, nothing
    end

end