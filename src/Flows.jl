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

Ind{N} = Union{Integer, NTuple{N, Integer}}

@doc raw""" 

    sample([rng=default_rng(), ] flow, dims [, θ = dflt_θ(T, dims)] )

Sample the flow distribution.

Give a sample of the flow distribution of size given by `dims` which
can be an `Integer`, a `NTuple` integers. The default random sampler
is `Random.default_rng()` but any sampler of type `Random.AbstractRNG`
can be used. 

If the flow is conditional, `θ` can be passed as an array 
of size `(n, dims...)` where n is the number of parameters 
if `θ` different for at least two sampled points. Otherwise 
`θ` can be given as an `AbstractVector` or `Tuple{Vararg}` of 
length n, in that case all points are drawn with the same 
`θ` parameter.

# Example
```julia
# return a sample of size (20, 10)
sample(flow, (20, 10))

# return a sample of size (20, 10) at 
# parameter (1f0, 2f0) if flow is conditional
sample(flow, (20, 10), [1f0, 2f0])
```
"""
function sample(
    rng::Random.AbstractRNG,
    flow::Flow{T},
    dims::Ind{N},
    θ::AbstractArray{T} = dflt_θ(T, dims)
    ) where {T, N}

    d = flow.metadata.d
    r = reshape(rand(rng, flow.base, *(dims...)), (d, dims...))
    forward!(flow, r, θ)
    
    return r

end

sample(rng::Random.AbstractRNG, flow::Flow{T}, dims::Ind{N}, θ::AbstractVector{T}) where {T,N} = sample(rng, flow, dims, θ .* ones(1, dims))
sample(rng::Random.AbstractRNG, flow::Flow{T}, dims::Ind{N}, θ::Tuple{Vararg{T}}) where {T,N} = sample(rng, flow, dims, collect(θ))

sample(flow::Flow{T}, dims::Ind{N}, θ::AbstractArray{T} = dflt_θ(T, dims)) where {T,N} = sample(Random.default_rng(), flow, dims, θ)
sample(flow::Flow{T}, dims::Ind{N}, θ::AbstractVector{T}) where {T,N} = sample(Random.default_rng(), flow, dims, θ)
sample(flow::Flow{T}, dims::Ind{N}, θ::Tuple{Vararg{T}}) where {T,N} = sample(Random.default_rng(), flow, dims, θ)


@doc raw"""
    
    logpdf(flow, x [, θ = dflt_θ(x)])

Natural logarithm of the probability density function given by the flow.

Argument `x` can be given as an array of size (`d`, dims...) or as a Tuple
of `d` vectors of each length. In the latter case, return the logpdf on a
grid of values defined by these vectors.

If the flow is conditional, `θ` can be passed as an array 
of size `(n, dims...)` where n is the number of parameters 
if `θ` different for at least two sampled points. Otherwise 
`θ` can be given as an `AbstractVector` or `Tuple{Vararg}` of 
length n.

!!! warning
    If `x` is given as a `Tuple` of vectors, `θ` must be passed
    as an `AbstractVector` or `Tuple{Vararg}` of length n.

# Example
```julia
# Give arrays in the training range of the flow
x = range(1f0, 10f0, 40)
y = range(-2.5f0, 11f0, 10)
z = range(0.1f0, 2f0, 30)

# For a given trained Flow 'flow' on 3 dimensions
res = logpdf(flow, (x, y, z)) # if unconditional
res = logpdf(flow, (x, y, z), [1f0, 2f0]) # if 2 conditions

# using contour in Plots
contour(x, y, res[:, :, 1]')
contour(x, z, res[:, 4, :]')
```

See also [`pdf`](@ref).
"""
function logpdf(
    flow::Flow{T},
    x::AbstractArray{T},
    θ::AbstractArray{T}
    ) where {T}

    z, ln_det_jac = backward(flow, x, θ)
    return Distributions.logpdf(flow.base, z) .+ ln_det_jac

end

logpdf(flow::Flow{T}, x::AbstractArray{T}) where {T} = logpdf(flow, x, dflt_θ(x))
logpdf(flow::Flow{T}, x::AbstractArray{T}, θ::AbstractVector{T}) where {T} = logpdf(flow, x, θ .* ones(T, 1, size(x)[2:N]...))
logpdf(flow::Flow{T}, x::AbstractArray{T}, θ::Tuple{Vararg{T}}) where {T} = logpdf(flow, x, collect(θ))


function logpdf(
    flow::Flow{T}, 
    x::Tuple{Vararg{AbstractVector{T}}}, 
    θ::AbstractVector{T}
    ) where {T}
    
    lens = [length(v) for v in x]
    y = hcat(collect.(collect(Iterators.product(x...)))...) 
    t = θ .* ones(T, 1, *(lens...))
    return reshape(logpdf(flow, y, t), lens...)

end

logpdf(flow::Flow{T}, x::Tuple{Vararg{AbstractVector{T}}}) where {T} = logpdf(flow, x, T[])
logpdf(flow::Flow{T}, x::Tuple{Vararg{AbstractVector{T}}}, θ::Tuple{Vararg{T}})  where {T} = logpdf(flow, x, collect(θ))




@doc raw"""
    
    pdf(flow, x, [, θ = dflt_θ(x)])

Probability density function given by the flow.

See also [`logpdf`](@ref).
"""
pdf(flow::Flow, x, θ) = exp.(logpdf(flow, x, θ))
pdf(flow::Flow, x) = exp.(logpdf(flow, x))


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