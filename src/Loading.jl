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

export save, load


###########################################################
## Utils

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

is_atomic(element::Any) = false
is_atomic(::Type{T}) where {T<:Any} = false

get_type(::Type{T}) where {T} = string(Base.typename(T).wrapper)



#########################################################
# automatic save

@auto_save CouplingAxes
@auto_save NormalizationLayer
@auto_save MetaData


###########################################################
## Save / load simple Chain model

# Create a Dense Flux layer from a symbol to be interpreted as a function when evaluated
function Flux.Dense((in, out)::Pair{<:Integer, <:Integer}, σ::Symbol; init = Flux.glorot_uniform, bias = true)
    return Dense(init(out, in), bias, @eval (Flux.$σ))
end

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
## Save / load FlowElements


@doc raw"""
    
    save(directory, element; kws...)

Recursively save the [`FlowElement`](@ref) `element`'s weights in `directory`.

Setting `erase = true` force deletes any existing directory with the same name.
"""
function save(directory::String, element::T; erase::Bool = false) where {T<:FlowElement}

    make_dir(directory, erase = erase)
    filename = directory * "/" * get_type(T)

    # if the element itself is atomic simply save it
    if is_atomic(element) 
        _save(filename, element)
    else
        for field in fieldnames(T)

            value = getfield(element, field)

            # if the element is a FlowChain 
            # directly save the sub-elements of the array
            if element isa FlowChain
                for (il, sub_element) in enumerate(value)
                    
                    if (sub_element isa FlowElement) && is_atomic(sub_element)
                        # if the sub_element is atomic, save this element only in the correct directory
                        # here we call the function save specific to this sub_element
                        make_dir(filename * "_" * string(field) * "_@" * string(il))
                        _save(filename * "_" * string(field) * "_@" * string(il) * "/$(get_type(typeof(sub_element)))", sub_element)
                    else
                        # if the sub element is composite then call back save recursively
                        save(filename * "_" * string(field) * "_@" * string(il), sub_element)
                    end

                end
            else
                if is_atomic(value)
                    ## TO MODIFY HERE isa FlowEmlement
                    _save(filename * "_" * string(field), value)
                else
                    save(filename * "_" * string(field), value)
                end 
            end
        end
    end
end


function lookup_type(x::AbstractString)
    sym = Symbol(x)
    if isdefined(DensityFlows, sym)
        return getfield(DensityFlows, sym)
    elseif isdefined(Main, sym)
        return getfield(Main, sym)
    else
        error("Type $x not found in DensityFlows or Main")
    end
end

@doc raw""" 

    load(directory)

Load any [`FlowElement`](@ref) saved in `directory`.
"""
function load(directory::String)

    files = readdir(directory)
    elements = Any[]

    @assert length(files) > 0 "Needs to be at least one file / folder in the directory"
    
    # type of the object to be constructed
    str_file_1 = split(files[1], ".jld2")[1]
    str_type   = split(str_file_1, "_")[1]
    m_type = lookup_type(str_type)

    # if the type is atomic then we do not need to go further
    # we directly call the specific loading function
    if (m_type <: FlowElement) && is_atomic(m_type)
        return load(directory * "/" * str_file_1, m_type)
    end

    # if not atomic we collect all the fields

    # get the correct order of the fields
    field_order = fieldnames(m_type)
 
    # get all fields to load
    # Tuple(order_field, composite, order_array, array_of_filenames)
    # order_array: index of the element of the array read in that file (specified by @#)
    # order_field: index of the field associated to that file in the fieldname list
    fields = Dict{Symbol, Tuple{Int, Bool, Vector{Int}, Vector{String}}}()

    # first look to find all fields that need to be loaded
    for file in files

        # boolean value to know if we open composite element or not
        composite = true
        
        # get the field name by manipulating the filename
        str_field = file
        str_field = *([s * "_" for s in split(str_field, "_")[2:end]]...)[1:end-1]

        if length(split(str_field, ".")) > 1
            str_field = *([s * "." for s in split(str_field, ".")[1:end-1]]...)[1:end-1]
        end

        # position of the array (if not an array pos = 1 by default)
        # get the position using the key '@' in the file definition
        pos = 1

        if str_field[end-1] == '@'
            pos = tryparse(Int, string(str_field[end]))
            str_field = str_field[1:end-3]
        end

        # field that to be constructed in that part of the loop
        field = Symbol(str_field)

        # check if we reached a jld2 file or not
        # this could be improved in order to accept
        # other extension names
        m_file = file
        if length(file) > 4 && file[end-4:end] == ".jld2"
            composite = false
            m_file = file[1:end-5]
        end

        # populate the dictionnary
        if !haskey(fields, field)
            # initialise with the file values
            fields[field] = (findfirst(field_order .=== field), composite, Int[pos], String[directory * "/" * m_file])
        else
            # complete the order_array and array_of_filenames vectors
            push!(fields[field][3], pos)
            push!(fields[field][4], directory * "/" * m_file)
        end

    end

    # prepare a vector containing all loaded instances for every fields
    elements = Vector{Any}(undef, length(fields))

    # now go over the different fields
    for (field, data) in fields
        
        # get back all properties stored for a given field
        order_field, composite, order_array, filenames = data
        n_array = length(order_array)

        # if we do have to load an array
        if n_array > 1

            temp = Vector{Any}(undef, n_array)

            for k in 1:n_array
                filename = filenames[k]
                temp[order_array[k]] = !composite ? load(filename, fieldtype(m_type, field)) : load(filename)
            end
    
        else
            filename = filenames[1]
            temp = !composite ? load(filename, fieldtype(m_type, field)) : load(filename)
        end

        elements[order_field] = temp

    end

    return m_type(elements...)

end



###########################################################
## Save / load flow

function save(directory::String, flow::Flow; erase::Bool = false)

    make_dir(directory, erase = erase)
 
    for field in [:model, :metadata]
        save(directory * "/" * string(field), getfield(flow, field))
    end

    try
        JLD2.jldsave(directory * "losses.jld2"; Dict(field => getfield(flow, field)  for field in [:train_loss, :valid_loss])...)
    catch e
        println("Impossible to save $flow")
        rethrow(e)
    end

end