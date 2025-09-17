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
struct Flow{T, M<:FlowChain, D<:Distributions.Distribution, U<:AbstractArray{T}}
    
    model::M
    base::D

    metadata::MetaData{T, U}

    train_loss::Vector{T}
    valid_loss::Vector{T}

end

@auto_functor Flow


function summarize(obj::Flow)
    println("- model --------------------")
    summarize(obj.model)
    println("- base distribution --------")
    println("• " * string(Base.typename(typeof(obj.base)).wrapper))
end



function Flow(
    base::Distributions.Distribution, 
    model::FlowChain, 
    data::DataArrays{T}, 
    train_loss::Vector{T}, 
    valid_loss::Vector{T}
    ) where {T}

    # get the relevant information for the flow
    # and store them in a metadata attribute
    d = number_dimensions(data)
    n = number_conditions(data)
    θ_min = minimum_θ(data)
    θ_max = maximum_θ(data)

    metadata = MetaData("", d, n, θ_min, θ_max)
   
    return Flow(model, base, metadata, train_loss, valid_loss)

end

function Flow(
    model::FlowChain, 
    data::DataArrays{T}, 
    train_loss::Vector{T}, 
    valid_loss::Vector{T}
    ) where {T}

    # get the relevant information for the flow
    # and store them in a metadata attribute
    d = number_dimensions(data)
    n = number_conditions(data)
    θ_min = minimum_θ(data)
    θ_max = maximum_θ(data)

    metadata = MetaData("", d, n, θ_min, θ_max)

    base = Distributions.MvNormal(zeros(T, d), LinearAlgebra.diagm(ones(T, d)))
   
    return Flow(model, base, metadata, train_loss, valid_loss)

end


Flow(base::Distributions.Distribution, model::FlowChain, data::DataArrays{T}) where {T} = Flow(base, model, data, T[], T[]) 
Flow(model::FlowChain, data::DataArrays{T}) where {T} = Flow(model, data, T[], T[]) 


# define a predict function that is the same as the forward function
predict(flow::Flow{T}, z::AbstractArray{T}, θ::AbstractArray{T}) where {T} = forward(flow, z, θ)[1]


function sample(
    flow::Flow{T},
    n::Int = 1,
    θ::AbstractArray{T} = dflt_θ(T, n),
    rng::Random.AbstractRNG = Random.default_rng()
    ) where {T}

    r = rand(rng, flow.base, n)
    forward!(flow, r, θ)
    
    return r

end


@doc raw"""
    
    logpdf(flow, x, θ = nothing)

Natural logarithm of the probability density function given by the flow.

See also [`pdf`](@ref).
"""
function logpdf(
    flow::Flow{T},
    x::AbstractArray{T},
    θ::AbstractArray{T}
    ) where {T}

    z, ln_det_jac = backward(flow, x, θ)
    return Distributions.logpdf(flow.base, z)  .+ ln_det_jac

end




@doc raw"""
    
    pdf(flow, x, θ = dflt_θ(x))

Probability density function given by the flow.

See also [`logpdf`](@ref).
"""
function pdf(
    flow::Flow{T},
    x::AbstractArray{T},
    θ::AbstractArray{T}
    ) where {T}

    return exp(logpdf(flow, x, θ))

end


@inline function loss(
    z::AbstractArray{T},
    ln_det_jac::AbstractArray{T},
    base::Distributions.Distribution,
    ) where {T}

    return - Distributions.mean(Distributions.logpdf(base, z) .+ ln_det_jac)
end


function train!(
    flow::Flow{T},
    data::DataArrays{T}, 
    optimiser_state::NamedTuple; 
    epochs::Int=100,
    batchsize=64,
    shuffle=true,
    verbose::Bool=true,
    debug::Bool=false
    ) where {T}
    
    train_data = training_data(data)
    valid_data = validation_data(data)
    
    train_loader = Flux.DataLoader(train_data; batchsize=batchsize, shuffle=shuffle)

    for _ ∈ 1:epochs
        
        for (x_batch, t_batch) ∈ train_loader

            grads = Flux.gradient(flow.model) do m
            
                z, ln_det_jac = backward(m, x_batch, t_batch)
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