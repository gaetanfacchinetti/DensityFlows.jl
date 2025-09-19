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

struct Flow{T, D, N, M<:FlowChain, B<:Distributions.Distribution, V<:AbstractVector{T}}
    
    model::M
    base::B

    metadata::MetaData{T, V}

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



@doc raw"""

    Flow([base, ] model, data)

Create a Flow from a model [`FlowChain`](@ref) and for a specific data.

The data must be passed as a [`DataArrays`](@ref). 
A spceific `base` distribution can be given, default is multivariate gaussian. 
"""
Flow

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
   
    return Flow{T, d, n, typeof(model), typeof(base), typeof(θ_min)}(model, base, metadata, train_loss, valid_loss)

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
   
    return Flow{T, d, n, typeof(model), typeof(base), typeof(θ_min)}(model, base, metadata, train_loss, valid_loss)

end


Flow(base::Distributions.Distribution, model::FlowChain, data::DataArrays{T}) where {T} = Flow(base, model, data, T[], T[]) 
Flow(model::FlowChain, data::DataArrays{T}) where {T} = Flow(model, data, T[], T[]) 


# define a predict function that is the same as the forward function
predict(flow::Flow{T}, z::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = forward(flow, z, θ)[1]

@doc raw""" 

    sample([rng=default_rng(), ] flow, dims [, θ = dflt_θ(T, dims)] )

Sample the flow distribution.

Give a sample of the flow distribution of size given by `dims` which
can be an `Integer`, or `NTuple` of integers. The default random sampler
is `Random.default_rng()` but any sampler of type `Random.AbstractRNG`
can be used. 

If the flow is conditional, `θ` can be passed as an array 
of size `(n, dims...)` where n is the number of parameters 
if `θ` different for at least two sampled points. Otherwise 
`θ` can be given as a `NTuple{n, T}`. In that case all points 
are drawn with the same `θ` parameter.

# Example
```julia
# return a sample of size (20, 10)
sample(flow, (20, 10))

# return a sample of size (20, 10) at 
# parameter (1f0, 2f0) if flow is conditional
sample(flow, (20, 10), (1f0, 2f0))
```
"""
function sample end

function sample(
    rng::Random.AbstractRNG,
    flow::Flow{T,D},
    dims::NTuple{M, Integer},
    θ::AbstractArray{T, K} = dflt_θ(T, dims)
    ) where {T,D,M,K}

    # need to remove this assert if in hot loop
    @assert K == M+1 "dimensions θ must match (n, dims...) with n number of trained parameters"

    r = reshape(rand(rng, flow.base, *(dims...)), (D, dims...))
    forward!(flow, r, θ)
    
    return r

end

function sample(
    rng::Random.AbstractRNG,
    flow::Flow{T,D,N},
    dims::Tuple{Vararg{T}},
    θ::NTuple{N, T}
    ) where {T,D,N}
    
    r = reshape(rand(rng, flow.base, *(dims...)), (D, dims...))
    forward!(flow, r, collect(θ) .* ones(1, dims...) )
    
    return r
end

sample(rng::Random.AbstractRNG, flow::Flow{T}, dims::Integer, θ::AbstractArray{T} = dflt_θ(T, dims)) where {T} = sample(rng, flow, (dims, ), θ)
sample(rng::Random.AbstractRNG, flow::Flow{T,D,N}, dims::Integer, θ::NTuple{N, T}) where {T,D,N} = sample(rng, flow, (dims, ), θ)


sample(flow::Flow{T}, dims::Union{Integer, Tuple{Vararg{Integer}}}, θ::AbstractArray{T} = dflt_θ(T, dims)) where {T} = sample(Random.default_rng(), flow, dims, θ)
sample(flow::Flow{T}, dims::Union{Integer, Tuple{Vararg{Integer}}}, θ::Tuple{Vararg{T}}) where {T} = sample(Random.default_rng(), flow, dims, θ)




@doc raw"""
    
    logpdf(flow, x [, θ = dflt_θ(x)])

Natural logarithm of the probability density function given by the flow.

Argument `x` can be given as an array of size (`d`, dims...) or as a Tuple
of `d` vectors of each length. In the latter case, return the logpdf on a
grid of values defined by these vectors.

If the flow is conditional, `θ` can be passed as an array 
of size `(n, dims...)` where n is the number of parameters 
if `θ` different for at least two sampled points. Otherwise 
`θ` can be given as a `NTuple{n, T}`.

!!! warning
    If `x` is given as a `Tuple` of vectors, 
    `θ` must be passed as a `NTuple{n, T}`.

# Example
```julia
# Give arrays in the training range of the flow
x = range(1f0, 10f0, 40)
y = range(-2.5f0, 11f0, 10)
z = range(0.1f0, 2f0, 30)

# For a given trained Flow 'flow' on 3 dimensions
res = logpdf(flow, (x, y, z)) # if unconditional
res = logpdf(flow, (x, y, z), (1f0, 2f0)) # if 2 conditions

# using contour in Plots
contour(x, y, res[:, :, 1]')
contour(x, z, res[:, 4, :]')
```

See also [`pdf`](@ref).
"""
function logpdf end


function logpdf(
    flow::Flow{T},
    x::AbstractArray{T,N},
    θ::AbstractArray{T,N}
    ) where {T,N}

    z, ln_det_jac = backward(flow, x, θ)
    return Distributions.logpdf(flow.base, z) .+ ln_det_jac

end

logpdf(flow::Flow{T}, x::AbstractArray{T}) where {T} = logpdf(flow, x, dflt_θ(x))
logpdf(flow::Flow{T}, x::AbstractArray{T,N}, θ::Tuple{Vararg{T}}) where {T,N} = logpdf(flow, x, collect(θ))


function logpdf(
    flow::Flow{T,D,N}, 
    x::NTuple{D, AbstractVector{T}}, 
    θ::NTuple{N, T}
    ) where {T,D,N}

    # array of the lengths of the arrays in x
    lens = [length(v) for v in x]

    # allocate the output in order to help compiler
    # with type inference
    res = Array{T, length(x)}(undef, lens...)
    
    # resshape the entire input into a (d, p) array
    # where p is the product of the length of all 
    # arrays in x. θ must also be of shape (d, p) then
    y = hcat(collect.(collect(Iterators.product(x...)))...) 
    res .= reshape(logpdf(flow, y, T.(collect(θ)) .* ones(T, 1, *(lens...))), lens...)

    return res

end


function logpdf(
    flow::Flow{T,D}, 
    x::NTuple{D, AbstractVector{T}}, 
    ) where {T,D}

    # array of the lengths of the arrays in x
    lens = [length(v) for v in x]

    # allocate the output in order to help compiler
    # with type inference
    res = Array{T, length(x)}(undef, lens...)
    
    # resshape the entire input into a (d, p) array
    # where p is the product of the length of all 
    # arrays in x. θ must also be of shape (d, p) then
    y = hcat(collect.(collect(Iterators.product(x...)))...) 
    res .= reshape(logpdf(flow, y, dflt_θ(y)), lens...)

    return res

end



@doc raw"""
    
    pdf(flow, x, [, θ = dflt_θ(x)])

Probability density function given by the flow.

See also [`logpdf`](@ref).
"""
function pdf end

pdf(flow::Flow{T}, x::AbstractArray{T,N}, θ::AbstractArray{T,N}) where {T,N} = exp.(logpdf(flow, x, θ))
pdf(flow::Flow{T}, x::AbstractArray{T}) where {T} = exp.(logpdf(flow, x))
pdf(flow::Flow{T}, x::AbstractArray{T,N}, θ::Tuple{Vararg{T}}) where {T,N} = exp.(logpdf(flow, x, θ))
pdf(flow::Flow{T,D,N}, x::NTuple{D, AbstractVector{T}}, θ::NTuple{N, T}) where {T,D,N} = exp.(logpdf(flow, x, θ))
pdf(flow::Flow{T,D}, x::NTuple{D, AbstractVector{T}}) where {T,D} = exp.(logpdf(flow, x))


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
    
    train_data = normalized_training_data(data, flow.metadata)
    valid_data = normalized_validation_data(data, flow.metadata)
    
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