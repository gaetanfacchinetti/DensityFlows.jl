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
# Contains some wrapping functions of Flux for convinience
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################


export FluxChain

# wrapper of Flux.Chain
struct FluxChain{T, C<:Flux.Chain} 
    net::C
end

Flux.@layer FluxChain trainable=(net,)

function FluxChain(chain::Flux.Chain) 
    return FluxChain{Float32, typeof(chain)}(chain)
end

(f::FluxChain{T})(x::AbstractArray{T}) where {T} = f.net(x)

FluxChain(xs::Tuple) = FluxChain(Flux.Chain(xs))
FluxChain(xs::AbstractVector) = FluxChain(Flux.Chain(xs))
FluxChain(xs::NamedTuple) = FluxChain(Flux.Chain(xs))
FluxChain(xs...) = FluxChain(Flux.Chain(xs...))

for fname ∈ (
    :(Base.getindex), 
    :(Base.length), 
    :(Base.first), 
    :(Base.last), 
    :(Base.iterate), 
    :(Base.lastindex),
    :(Base.firstindex),
    :(Base.keys),
    )

    @eval $fname(chain::FluxChain, args...; kws...) = $fname(chain.net, args...; kws...)
    
end