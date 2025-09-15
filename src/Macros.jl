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


export @auto_flow, @auto_forward, @auto_functor, @summary



macro _flowtrainable(T, fields)
    return esc(quote
        # Specify exactly what are the trainable parameters
        function Optimisers.trainable(m::$T)
            return (; (field => Optimisers.trainable(getfield(m, field)) for field in $fields)...)
        end
    end)
end

macro _flowlayer(T)
    return esc(quote
        Flux.@layer $T
        Functors.@functor $T
    end)
end

macro auto_flow(T)
    return esc(quote
        @_flowlayer $T
        @_flowtrainable $T fieldnames($T)
    end)
end


macro auto_flow(T, fields)
    return esc(quote
        @_flowlayer $T
        @_flowtrainable $T $fields
    end)
end


@doc raw"""

    auto_flow(T [, fields])

Automatically makes type `T` a `Flux` layer with trainable
parameters `fields` (array of `Symbols`). If no `fields` is passed 
all fields are assumed to be trainable if they can be. 
"""
macro auto_flow end


@doc raw"""

    auto_forward(T)

Automatically define a [`forward!`](@ref) function for type `T` 
from [`forward`](@ref) if there is no possible optimization to 
be found in  writting a specific [`forward!`](@ref) function.
"""
macro auto_forward(T)
    
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


macro summary(obj)
    return :(_print($(esc(obj))))
end