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

export train!, loss

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

@inline function max_likelihood_loss(
    z::AbstractArray{T},
    ln_det_jac::AbstractArray{T},
    base::Distributions.Distribution,
    ) where {T}

    return - Distributions.mean(Distributions.logpdf(base, z) .+ ln_det_jac)
end


function _noise_generator(generator::Function, metadata::MetaData{T}, x::AbstractArray{T}, t::AbstractArray{T}) where {T} 
    res = generator(x, resize_output(t, metadata.θ_min, metadata.θ_max))
    return res[1], normalize_input(res[2], metadata.θ_min, metadata.θ_max)
end


function update_info!(
    loss::Function,
    flow::Flow{T}, 
    train_data::NTuple{2, AbstractArray{T}}, 
    valid_data::NTuple{2, AbstractArray{T}},
    verbose::Bool
    ) where {T}

    z, ln_det_jac = backward(flow.model, train_data...)
    train_loss = loss(z, ln_det_jac, flow.base)
    
    ## Training loss
    if (train_loss != train_loss) || isinf(train_loss)
        error = TrainingError(z, ln_det_jac, train_loss, "Error on the training loss")
        throw(error)
    end

    # proceed with the update
    push!(flow.train_loss, train_loss)

    ## Validation loss
    z, ln_det_jac = backward(flow.model, valid_data...)
    valid_loss = loss(z, ln_det_jac, flow.base)
    
    if (valid_loss != valid_loss) || isinf(valid_loss)
        error = TrainingError(z, ln_det_jac, valid_loss, "Error on the validation loss")
        throw(error)
    end

    # proceed with the update
    push!(flow.valid_loss, valid_loss)

    # if everything went well print the updated info
    if verbose 
        println("epoch: $(length(flow.train_loss)) | train_loss = $train_loss, valid_loss = $valid_loss")
    end

end



function train!(
    loss::Function,
    flow::Flow{T},
    train_data::NTuple{2, AbstractArray{T}},
    valid_data::NTuple{2, AbstractArray{T}},
    optimiser_state::NamedTuple; 
    epochs::Int=100,
    batchsize::Int=64,
    shuffle::Bool=true,
    verbose::Bool=true,
    noise_generator::Function=(x, θ)->(x, θ),
    ) where {T}
        
    train_loader = Flux.DataLoader(train_data; batchsize=batchsize, shuffle=shuffle)

    try

        for _ ∈ 1:epochs
            
            for (x_batch, t_batch) ∈ train_loader

                grads = Flux.gradient(flow.model) do m
                
                    z, ln_det_jac = backward(m, _noise_generator(noise_generator, flow.metadata, x_batch, t_batch)...)#x_batch, t_batch_noise)
                    l = loss(z, ln_det_jac, flow.base)

                    # debugging if loss gets NaN
                    ((l != l) || isinf(l)) && throw(TrainingError(z, ln_det_jac, l, "Error during training"))

                    return l 
                                    
                end

                # take one step in the stochastic gradient descent
                Optimisers.update!(optimiser_state, flow.model, grads[1])

            end
            
            # update training and validation loss records
            update_info!(loss, flow, train_data, valid_data, verbose)
    
        end

    catch e
        isa(e, TrainingError) && (print_training_error(e); return nothing)
        rethrow(e)
    end

end


@doc raw"""

    train!(flow, data, state; kws...)

Train the flow to match the input distribution.

# Arguments
- `flow::Flow{T}`: flow to train
- `data::DataArrays{T}`: data to train on
- `state::NamedTuple`: optimiser state

# Keyword arguments
- `epochs::Int`: number of epochs, default is 100
- `batchsize::Int`: size of each batch, default is 64
- `shuffle::Bool`: if true, shuffle the data in batches, default is true
- `verbose::Bool`: if true, output the current state, default is true
"""
function train!(
    loss::Function,
    flow::Flow{T}, 
    data::DataArrays{T}, 
    optimiser_state::NamedTuple; 
    kws...
    ) where {T}

    train_data = normalized_training_data(data, flow.metadata)
    valid_data = normalized_validation_data(data, flow.metadata)
    
    train!(loss, flow, train_data, valid_data, optimiser_state; kws...)

end


function train!(
    flow::Flow{T}, 
    data::DataArrays{T}, 
    optimiser_state::NamedTuple; 
    kws...
    ) where {T}

    train!(max_likelihood_loss, flow, data, optimiser_state; kws...)

end
