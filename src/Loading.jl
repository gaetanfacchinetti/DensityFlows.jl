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
# Contains functions for saving and loading the models

# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################



###########################################################
## Utils


function add_suffix(filename::String, suffix::String)
    name_arr = split(filename, ".")
    name_bulk, name_ext = name_arr[1:end-1], name_arr[end]
    return *([p * "." for p in name_bulk]...)[1:end-1] * "_" * suffix * "." * name_ext
end

function remove_extension(filename::String)
    name_arr = split(filename, ".")
    return *([p * "." for p in name_arr[1:end-1]]...)[1:end-1]
end


function make_dir(path::AbstractString; erase::Bool = false)
    
    if isdir(path)
        
        if erase
            # remove existing directory
            rm(path; recursive=true)    
        else
            throw(ArgumentError("Impossible to save at $path if erase is false as a folder of the same name already exists."))
        end

    end

    # create fresh directory
    mkpath(path)   
                       
end

get_type(::Type{T}) where {T<:AffineCouplingBlock} = "AffineCouplingBlock"
get_type(::Type{T}) where {T<:RNVPCouplingLayer} = "RNVPCouplingLayer"
get_type(::Type{T}) where {T<:NICECouplingLayer} = "NICECouplingLayer"

get_type(::Type{T}) where {T<:Flux.Chain} = "Chain"
get_type(::Type{T}) where {T<:AffineCouplingAxes} = "AffineCouplingAxes"


###########################################################
## Save / load AffineCouplingAxes

function save(filename::String, axes::AffineCouplingAxes)

    try
        JLD2.jldsave(filename * ".jld2"; Dict(field => getfield(axes, field)  for field in fieldnames(AffineCouplingAxes))...)
    catch e
        println("Impossible to save the axes")
        rethrow(e)
    end

end

function load(filename::String, ::Type{AffineCouplingAxes})

    try
        data = JLD2.jldopen(filename * ".jld2")
        return AffineCouplingAxes([data[k] for k in string.(fieldnames(AffineCouplingAxes))]...)
    catch e
        println("Impossible to load type $T at $filename")
        rethrow(e)
    end

end


# Create a Dense Flux layer from a symbol to be interpreted as a function when evaluated
function Flux.Dense((in, out)::Pair{<:Integer, <:Integer}, σ::Symbol; init = Flux.glorot_uniform, bias = true)
    return Dense(init(out, in), bias, @eval (Flux.$σ))
end


###########################################################
## Save / load simple Chain model


function save(filename::String, model::Flux.Chain)

    # save the structure parameter of the layer and the value of the trainable parameters
    
    inputs = [size(l.weight, 2) for l in model.layers]
    outputs = [size(l.weight, 1) for l in model.layers]
    activations = [string(Symbol(l.σ)) for l in model.layers]
    bias = [l.bias for l in model.layers]
    
    state = Flux.state(model)

    try
        JLD2.jldsave(filename * ".jld2"; inputs, outputs, activations, bias, state)
    catch e
        println("Impossible to save the the RNVPCouplingLayer")
        rethrow(e)
    end

end


function load(filename::String, ::Type{T}) where {T<:Flux.Chain}

    try
        data = JLD2.jldopen(filename * ".jld2")
        model = Flux.Chain([Flux.Dense(data["inputs"][i], data["outputs"][i], Symbol(data["activations"][i]), bias = data["bias"][i]) for i in 1:length(data["inputs"])])
        Flux.loadmodel!(model, data["state"])
    catch e
        println("Impossible to load the chain at $filename")
        rethrow(e)
    end
end


###########################################################
## Save / load AffineCouplingLayer, AffineCouplingBlock, AffineCouplingChain model

function save(directory::String, element::T; erase::Bool = false) where {T<:Union{AffineCouplingLayer, AffineCouplingBlock}}

    make_dir(directory, erase = erase)
    filename = directory * "/" * get_type(T) * "_"

    for field in fieldnames(T)
        save(filename * string(field), getfield(element, field))
    end

end

function save(directory::String, chain::AffineCouplingChain; erase::Bool = false)

    make_dir(directory, erase = erase)
    filename = directory * "/AffineCouplingChain_"

    for (il, layer) in enumerate(chain.layers)
        save(filename * "layer_" * string(il), layer)
    end

end


function load(directory::String, ::Type{T}) where {T<:AffineCouplingLayer}
    
    file = readdir(directory)[1]

    if file[1:4] == "RNVP"
        filename = directory * "/RNVPCouplingLayer_"
        return RNVPCouplingLayer([load(filename * string(field), fieldtype(RNVPCouplingLayer, field)) for field in fieldnames(RNVPCouplingLayer)]...)
    elseif file[1:4] == "NICE"
        filename = directory * "/NICECouplingLayer_"
        return NICECouplingLayer([load(filename * string(field), fieldtype(NICECouplingLayer, field)) for field in fieldnames(NICECouplingLayer)]...)
    else
        throw(ArgumentError("Unknown type of AffineCouplingLayer"))
    end
end


function load(directory::String, ::Type{AffineCouplingBlock})
    filename = directory * "/AffineCouplingBlock_"
    return AffineCouplingBlock([load(filename * string(field), fieldtype(AffineCouplingBlock, field)) for field in fieldnames(AffineCouplingBlock)]...)
end

