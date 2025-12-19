##################################################################################
# This file is part of DensityFlows.jl
#
# Copyright (c) 2025, Gaétan Facchinetti
#
# jl is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. jl is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
##################################################################################
#
# Contains functions related to conditional masked autoregressive flows
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export MaskedDense, CMAFCouplingLayer

# ============================================================
# 1. Masked Dense Layer (MADE building block)
# ============================================================

struct MaskedDense{T, W<:AbstractMatrix{T}, B<:AbstractVector{T}, M<:AbstractMatrix{T}, F}
    w::W     # weight matrix
    b::B     # bias
    mask::M  # same size as the weight matrix
    σ::F     # activation function
end

# Make it compatible with Flux parameters
Flux.@layer MaskedDense trainable=(w, b)

function MaskedDense(
    ::Type{T}, 
    in::Int, 
    out::Int, 
    mask::AbstractMatrix{T}; 
    σ = Flux.identity, 
    init = Flux.glorot_uniform
    ) where {T}
    
    w = T.(init(out, in))
    b = zeros(T, out)#T(0.5) * rand(T, out)

    @assert size(w) == size(mask) "mask and weight sizes must match"
    return MaskedDense(w, b, mask, σ)
end

MaskedDense(in::Int, out::Int, args...; kws...) = MaskedDense(Float32, in, out, args...; kws...)

(m::MaskedDense{T})(x::AbstractArray{T}) where {T} = Flux.NNlib.bias_act!(m.σ, m.w .* m.mask * x, m.b)



# ============================================================
# 2. Conditional MAF layer
# ============================================================

function assign_degrees(width, d)
    @assert width >= d "number of neurons in each layer should be ≥ dimension"

    # make sure that every dimesion appear at least once in the degrees table
    return Random.shuffle!(vcat(collect(1:d), Random.rand(1:d, width-d)))
    #return shuffle!(repeat(collect(1:d), ceil(Int, width/d))[1:width])
end

struct CMAFCouplingLayer{T, S<:Union{Tuple{Vararg{<:MaskedDense{T}}}, AbstractVector{<:MaskedDense{T}}}, V<:AbstractVector{Int}} <: CouplingLayer
        
    d::Int
    n::Int
    
    sub_layers::S
    
    dim2deg::V
    deg2dim::V

end

function CMAFCouplingLayer(
    d::Int,
    n::Int,
    xs::Union{Tuple{Vararg{<:MaskedDense{T}}}, AbstractVector{<:MaskedDense{T}}},
    dim2deg::AbstractVector{Int},
    deg2dim::AbstractVector{Int},
    ) where {T} 
    
    # get the properties of the coupling layer from the sub_layers passed in input
    return CMAFCouplingLayer{T, typeof(xs), typeof(dim2deg)}(d, n, xs, dim2deg, deg2dim)
end

Flux.@layer CMAFCouplingLayer trainable=(sub_layers)

function CMAFCouplingLayer(
    ::Type{T}, 
    d::Int, 
    n::Int,
    dim2deg::AbstractVector{Int} = Random.randperm(d); 
    hidden::AbstractVector{Int} = [64,64],
    σ::Function = Flux.relu
    ) where {T}
    

    # --------------  autoregressive degrees --------------
    m = length(hidden)

    deg2dim = similar(dim2deg)

    for i in 1:d
        res = findall(i .== dim2deg)
        @assert length(res) == 1 "res = $res"
        deg2dim[i] = res[1]
    end

    degrees = Vector{Vector{Int}}(undef, m+2)

    # degrees for input and output
    degrees[1] = dim2deg
    degrees[end] = vcat(dim2deg .- 1, dim2deg .- 1)
    degrees[2:end-1] .= [assign_degrees(h, d) for h in hidden]

    # function to construct binary mask from degrees
    make_mask(a, b) = T.(a' .<= b)

    # masks
    masks = Vector{Matrix{T}}(undef, m+1)

    # the first mask should not mask the parameters
    masks[1] = make_mask(degrees[1], degrees[2])
    masks[1] = hcat(ones(hidden[1], n), masks[1])

    # set the rest of the mask
    masks[2:end] .= [make_mask(degrees[i], degrees[i+1]) for i in 2:(m+1)]


    # --------------  Build layers --------------
    sub_layers = Vector{MaskedDense{T}}(undef, m+1)

    # First masked layer
    sub_layers[1] = MaskedDense(d+n, hidden[1], masks[1], σ = σ)
    sub_layers[2:end-1] .= [MaskedDense(hidden[i-1], hidden[i], masks[i], σ = σ) for i in 2:m]
    sub_layers[end] = MaskedDense(hidden[end], 2*d, masks[end], σ = identity)

    return CMAFCouplingLayer{T, typeof(sub_layers), typeof(dim2deg)}(d, n, sub_layers, dim2deg, deg2dim)

end


CMAFCouplingLayer(d::Int, n::Int, dim2deg::AbstractVector{Int} = Random.randperm(d), args...; kws...) = CMAFCouplingLayer(Float32, d, n, dim2deg, args...; kws...)

function (f::CMAFCouplingLayer{T})(
    x::AbstractArray{T}, 
    θ::AbstractArray{T},
    dim::Union{Int, AbstractVector{Int}},
    ) where {T}

    z = vcat(θ, x)

    for layer in f.sub_layers
        z = layer(z)
    end

    μ = selectdim(z, 1, dim)
    s = selectdim(z, 1, dim .+ f.d)

    return (μ, s)

end

(f::CMAFCouplingLayer{T})(x::AbstractArray{T}, θ::AbstractArray{T}) where {T} = f(x, θ, 1:f.d)


function backward(
    layer::CMAFCouplingLayer{T}, 
    x::AbstractArray{T}, 
    θ::AbstractArray{T}
    ) where {T}

    μ, s = layer(x, θ)
    s = 5 * Flux.tanh.(s / 5)
    
    # ln|det J[T^{-1}]|
    ln_det_jac = -dropdims(sum(s, dims = 1), dims = 1)

    return (x .- μ) .* exp.(-s), ln_det_jac
end


function forward(
    layer::CMAFCouplingLayer{T},
    z::AbstractArray{T,N},
    θ::AbstractArray{T,N}
    ) where {T,N}

    x = zeros(T, size(z)...)
    ln_det_jac = zeros(T, size(z)[2:end]...)

    # initialize the first step with z
    dim = layer.deg2dim[1]
    μ, _s = layer(z, θ, dim)
    _s = 5 * Flux.tanh.(_s / 5)
    selectdim(x, 1, dim) .= μ .+ selectdim(z, 1, dim).*exp.(_s)
    ln_det_jac .= ln_det_jac .+ _s


    # reconstruct from the lowest degree to the highest
    for i in 2:layer.d
        dim = layer.deg2dim[i]
        μ, _s = layer(x, θ, dim)
        _s = 5 * Flux.tanh.(_s / 5)
        selectdim(x, 1, dim) .= μ .+ selectdim(z, 1, dim).*exp.(_s)
        ln_det_jac .= ln_det_jac .+ _s

    end

    return x, ln_det_jac

end


function forward!(
    layer::CMAFCouplingLayer{T},
    z::AbstractArray{T,N},
    θ::AbstractArray{T,N}
    ) where {T,N}

    #x = zeros(T, size(z)...)
    ln_det_jac = zeros(T, size(z)[2:end]...)

    # initialize the first step with z
    dim = layer.deg2dim[1]
    μ, _s = layer(z, θ, dim)
    _s = 5 * Flux.tanh.(_s / 5)
    selectdim(z, 1, dim) .= μ .+ selectdim(z, 1, dim).*exp.(_s)
    ln_det_jac .= ln_det_jac .+ _s

    # reconstruct from the lowest degree to the highest
    for i in 2:layer.d
        dim = layer.deg2dim[i]
        μ, _s = layer(z, θ, dim)
        _s = 5 * Flux.tanh.(_s / 5)
        selectdim(z, 1, dim) .= μ .+ selectdim(z, 1, dim).*exp.(_s)
        ln_det_jac .= ln_det_jac .+ _s
    end

    return nothing

end
