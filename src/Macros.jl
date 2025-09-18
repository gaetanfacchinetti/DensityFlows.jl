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
# Contains macros to create normalizing flows
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


export @auto_forward!, @auto_functor, @summary
export @save_as_atomic
export @save_element, @clear_and_save_element
export @save_flow, @clear_and_save_flow
export @load_element, @load_flow


@doc raw"""

    auto_forward!(T)

Automatically define a [`forward!`](@ref) function for type `T` 
from [`forward`](@ref) if there is no possible optimization to 
be found in  writting a specific [`forward!`](@ref) function.
"""
macro auto_forward!(T)
    
    return esc(quote
       
        function forward!(
            m::$T, 
            z::AbstractArray{U}, 
            θ::AbstractArray{U} = dflt_θ(z)
            ) where {U}

            z = forward(m, z, θ)
            return nothing
    
        end
    end)

end

@doc raw"""

    auto_functor(T)

Automatically define a functor function for type `T`
calling [`forward`](@ref).
"""
macro auto_functor(T)
    
    return esc(quote
        # Specify exactly what are the trainable parameters
        function (f::$T)(
            z::AbstractArray{U}, 
            θ::AbstractArray{U} = dflt_θ(z)
            ) where {U}

            return forward(f, z, θ)
    
        end
    end)

end


@doc raw"""

    auto_functor(element)

Call [`summarize`](@ref)(`element`).
"""
macro summary(obj)
    return :(summarize($(esc(obj))))
end



macro flow_wrapper(funcs...)
    return esc(Expr(:block, 
    [quote
        #$f(flow::Flow, y::AbstractArray) = $f(flow.model, y)
        function $f(flow::Flow, y::AbstractArray{T}, θ::AbstractArray{T}) where {T}
            $f(flow.model, y, normalize_input(θ, flow.metadata.θ_min, flow.metadata.θ_max))
        end
    end 
    for f in funcs]...))
end


@doc raw"""

    unconditional_wrapper(funcs...)

Define the unconditional version of a function with signature
f(::Flow, ::AbstractArray, ::AbstractArray) or
f(::FlowElement, ::AbstractArray, ::AbstractArray).

Replace f(obj, y, θ) = f(obj, y, dflt_θ(y))
"""
macro unconditional_wrapper(funcs...)
    return esc(Expr(:block, [quote $f(obj::Union{FlowElement, Flow{T}}, y::AbstractArray{T}) where {T} = $f(obj, y, dflt_θ(y)) end for f in funcs]...))
end

# default behaviour of the functions in DensityFlows
save_element_atomic() = nothing
load_element() = nothing

macro save_as_atomic(T)
    
    return esc(

        quote 
            
            function DensityFlows.save_element_atomic(filename::String, obj::$T)
                try
                    JLD2.jldsave(filename * ".jld2"; Dict(field => getfield(obj, field) for field in fieldnames($T))...)
                catch e
                    println("Impossible to save $obj")
                    rethrow(e)
                end
            end

            function DensityFlows.load_element(filename::String, ::Type{U}) where {U<:$T}
                try
                    data = JLD2.jldopen(filename * ".jld2")
                    return U([data[k] for k in string.(fieldnames(U))]...)
                catch e
                    println("Impossible to load $U at $filename")
                    rethrow(e)
                end
            end

            DensityFlows.is_atomic(element::$T) = true
            DensityFlows.is_atomic(::Type{$T}) = true

        end
        )
        
end


macro clear_and_save_element(filename, model)
    return esc(quote save_element($filename, $model, erase = true) end)
end

macro save_element(filename, model)
    return esc(quote save_element($filename, $model, erase = false) end)
end

macro load_element(filename)
    return esc(quote load_element($filename) end)
end

macro clear_and_save_flow(filename, flow)
    return esc(quote save_flow($filename, $flow, erase = true) end)
end

macro save_flow(filename, flow)
    return esc(quote save_flow($filename, $flow, erase = false) end)
end


macro load_flow(filename)
    return esc(quote load_flow($filename) end)
end




#macro select_trainables(T, fields)
#    return esc(quote
#        # Specify exactly what are the trainable parameters
#        function Optimisers.trainable(m::$T)
#            return (; (field => Optimisers.trainable(getfield(m, field)) for field in $fields)...)
#        end
#    end)
#end

# macro _flowlayer(T)
#     return esc(quote
#         Flux.@layer $T
#         Functors.@functor $T
#     end)
# end

# macro auto_flow(T)
#     return esc(quote
#         @_flowlayer $T
#         @select_trainables $T fieldnames($T)
#     end)
# end


# macro auto_flow(T, fields)
#     return esc(quote
#         @_flowlayer $T
#         @select_trainables $T $fields
#     end)
# end


# @doc raw"""

#     auto_flow(T [, fields])

# Automatically makes type `T` a `Flux` layer with trainable
# parameters `fields` (array of `Symbols`). If no `fields` is passed 
# all fields are assumed to be trainable if they can be. 
# """
# macro auto_flow end