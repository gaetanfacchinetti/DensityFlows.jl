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
# Contains functions related to the training of the flow
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export train!, loss, NFLoss, LogPdfLoss

abstract type NFLoss{T} end

struct TrainingError{T, A<:AbstractArray{T}, V<:AbstractArray{T}} <: Exception 
    
    z::A
    ln_det_jac::V
    loss::T

    msg::String

end

function print_training_error(e::TrainingError)
    println(e.msg)
    println("loss = $(e.loss)")
    println("z = $(e.z)")
    println("ln|J| = $(e.ln_det_jac)")
end

@doc raw"""

    LogPdfLoss(base, λ_reg)

Return a loss function structure.

# Arguments
- `base::Distributions.Distribution`: base distribution (must match that of the flow).
- `λ_reg::T`: coefficient of the regulator λ_reg E[log|det J|^2] in the loss function, default is zero.
"""
struct LogPdfLoss{T, B<:Distribution{T}} <: NFLoss{T}
    base::B
    λ_reg::T
end

function (f::LogPdfLoss)(
    z::AbstractArray{T},
    ln_det_jac::AbstractArray{T}
    ) where {T}

    return - Statistics.mean(logpdf(f.base, z) .+ ln_det_jac .- f.λ_reg * ln_det_jac.^2 )
end


function update_info!(
    loss::NFLoss{T},
    flow::Flow{T}, 
    train_data::NTuple{2, AbstractArray{T}}, 
    valid_data::NTuple{2, AbstractArray{T}},
    verbose::Bool
    ) where {T}

    ## Training loss
    x, θ = train_data
    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max)
    z, ln_det_jac = backward(flow.model, x, t)
    train_loss = loss(z, ln_det_jac)
    
    if (train_loss != train_loss) || isinf(train_loss)
        error = TrainingError(z, ln_det_jac, train_loss, "Error on the training loss")
        throw(error)
    end

    # proceed with the update
    push!(flow.train_loss, train_loss)

    # ---------------------------------

    ## Validation loss
    x, θ = valid_data
    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max)
    z, ln_det_jac = backward(flow.model, x, t)
    valid_loss = loss(z, ln_det_jac)
    
    if (valid_loss != valid_loss) || isinf(valid_loss)
        error = TrainingError(z, ln_det_jac, valid_loss, "Error on the validation loss")
        throw(error)
    end

    # proceed with the update
    push!(flow.valid_loss, valid_loss)

    # ---------------------------------

    # if everything went well print the updated info
    if verbose 
        println("epoch: $(length(flow.train_loss)) | train_loss = $train_loss, valid_loss = $valid_loss")
    end

end


@doc raw"""

    train!([noise, ], [loss, ] flow, data, state; kws...)

Train the flow to match the input distribution.

# Arguments
- `flow::Flow{T}`: flow to train.
- `data::DataArrays{T}`: data to train on.
- `state::NamedTuple`: optimiser state.
- `noise::NoiseModel{T}`: noise model injected during training.
- `loss::NFLoss{T}`: custom loss struct.

# Keyword arguments
- `epochs::Int`: number of epochs, default is 100.
- `batchsize::Int`: size of each batch, default is 64.
- `shuffle::Bool`: if true, shuffle the data in batches, default is true.
- `verbose::Bool`: if true, output the current state, default is true.

!!! info
    A NFLoss struct S should be associated with a functor 
    `(f::S)(z::AbstractArray{T}, lndj::AbstractArray{T}) where {T}``
    returning the value of the loss.
"""
function train! end


function train!(
    noise::NoiseModel{T},
    loss::NFLoss{T},
    flow::Flow{T},
    data::DataArrays{T},
    optimiser_state::NamedTuple; 
    epochs::Int=100,
    batchsize::Int=64,
    shuffle::Bool=true,
    verbose::Bool=true,
    ) where {T}
        


    # get the training and validation data
    train_data = training_data(data)
    valid_data = validation_data(data)

    # partition the data in shuffled batches
    train_loader = Flux.DataLoader(train_data; batchsize=batchsize, shuffle=shuffle)

    try

        for _ ∈ 1:epochs

            # put the model in train mode
            Flux.trainmode!(flow.model)

            
            for (x_batch, θ_batch) ∈ train_loader

                grads = Flux.gradient(flow.model) do m
                
                    # generate noisy data from the noise model
                    x, θ = noisy_data(noise, x_batch, θ_batch)
                    t = normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max)
                    
                    # compute the loss from the data
                    z, ln_det_jac = backward(m, x, t)
                    l = loss(z, ln_det_jac)#, flow.base, λ_reg)

                    # debugging if loss gets NaN
                    ((l != l) || isinf(l)) && throw(TrainingError(z, ln_det_jac, l, "Error during training"))

                    return l 
                                    
                end

                # take one step in the stochastic gradient descent
                Optimisers.update!(optimiser_state, flow.model, grads[1])

            end

            # put the model back into test mode
            Flux.testmode!(flow.model)
            
            # update training and validation loss records
            update_info!(loss, flow, train_data, valid_data, verbose)
    
        end

    catch e
        isa(e, TrainingError) && (print_training_error(e); return nothing)
        Flux.testmode!(flow.model)
        rethrow(e)
    end


end



function train!(
    loss::NFLoss{T},
    flow::Flow{T}, 
    args...; 
    kws...
    ) where {T}


    train!(NoNoise(T), loss, flow, args...; kws...)

end


function train!(
    noise::NoiseModel{T},
    flow::Flow{T}, 
    args...; 
    λ_reg::T = zero(T),
    kws...
    ) where {T}


    loss = LogPdfLoss(flow.base, λ_reg)
    train!(noise, loss, flow, args...; kws...)

end


train!(flow::Flow{T}, args...; λ_reg::T = zero(T), kws...) where {T} = train!(NoNoise(T), LogPdfLoss(flow.base, λ_reg), flow, args...; kws...)

