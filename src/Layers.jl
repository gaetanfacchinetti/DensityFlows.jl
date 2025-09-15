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
# Contains functions related to affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################




##################################################################################
# Layers


function _dflt_net(
    input_dim::Int,
    output_dim::Int,
    n::Int;
    hidden_dim::Int = 32,
    σ::Function = Flux.relu,
    bias::Bool = true)

        
    sublayers = (
        [Dense(input_dim, hidden_dim, σ, bias = bias)],
        [Dense(hidden_dim, hidden_dim, σ, bias = bias) for _ in 1:(n-1)],
        [Dense(hidden_dim, output_dim, Flux.identity, bias = bias)]
    )
    
    return Flux.Chain(vcat(sublayers...)...)

end



@doc raw"""
    
    CouplingLayer([T=RNVPCouplingLayer, ] axes; kws...)
    CouplingLayer([T=RNVPCouplingLayer, ] d, j = d ÷ 2; n=0, reverse=false, kws...)
    CouplingLayer([T=RNVPCouplingLayer, ] d, mask; n=0, kws...)
    
    CouplingLayer(t_net, axes)
    CouplingLayer(s_net, t_net, axes)
      
Create an CouplingLayer with NN models `s` and `t`.

The layer can be represented as a function `f`
such that on dimensions where it does not act 
like the identity it returns

```math
    f(x) = x * \exp(s) + t \quad {\rm if~forward}
```

and

```math
    f^{-1}(z) = \exp(-s) * (z-t) \quad {\rm if~backward} \, .
```

By default `s` and `t` are built with `Dense` neural networks.

# Arguments
- `axes::CouplingAxes`.
- `d::Int`: dimension of the flow.
- `j::Int`: dimension cut (default is `d`÷2).
- `mask::Vector{Int}`: dimensions that are affected by the coupling.

# Keywords arguments
- `n::Int`: number of conditions / parameters (default is 0).
- `hidden_dim::Int`: number of hidden dimensions in `s` and `t` (default is 32).
- `n_sublayers_t::Int`: number of sublayers in `t` (default is 2).
- `n_sublayers_s::Int`: number of sublayers in `s` (default is 2).
- `σ::Function`: activation function (default is `Flux.relu`).
- `bias::Bool`: activate bias (default is `true`).

# Example
```jldoctest
julia> @summary CouplingLayer(3, [1, 3], n=2, hidden_dim=10, n_sublayers_s=1, σ=Flux.tanh)
• RNVPCouplingLayer > s_net: [3, 10, 2] (62 parameters)
• RNVPCouplingLayer > t_net: [3, 10, 10, 2] (172 parameters)
• RNVPCouplingLayer > axes: (d,n)=(3,2), id=[2], af=[1, 3]
```

See also [`CouplingAxes`](@ref).
"""
CouplingLayer


CouplingLayer(t_net::Flux.Chain, axes::CouplingAxes) = NICECouplingLayer(t_net, axes)
CouplingLayer(s_net::Flux.Chain, t_net::Flux.Chain, axes::CouplingAxes) = RNVPCouplingLayer(s_net, t_net, axes)

function CouplingLayer(
    ::Type{T},
    axes::CouplingAxes;
    n_sublayers_t::Int = 2, 
    n_sublayers_s::Int = 2,
    kws...
    ) where {T<:CouplingLayer}
 

    input_dim  = length(axes.axis_nn)
    output_dim = length(axes.axis_af)

    t_net = _dflt_net(input_dim, output_dim, n_sublayers_t; kws...)
    (T === NICECouplingLayer) && return T(t_net, axes)
    
    s_net = _dflt_net(input_dim, output_dim, n_sublayers_s; kws...)

    return T(s_net, t_net, axes)

end


CouplingLayer(axes::CouplingAxes; kws...) = CouplingLayer(RNVPCouplingLayer, axes; kws...)
CouplingLayer(d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) = CouplingLayer(CouplingAxes(d, j, n=n, reverse=reverse); kws...)
CouplingLayer(d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) = CouplingLayer(CouplingAxes(d, mask, n=n); kws...)
CouplingLayer(::Type{T}, d::Int, j::Int = d ÷ 2; n::Int = 0, reverse::Bool = false, kws...) where {T<:CouplingLayer} = CouplingLayer(CouplingAxes(d, j, n=n, reverse=reverse); kws...)
CouplingLayer(::Type{T}, d::Int, mask::AbstractVector{Int}; n::Int = 0, kws...) where {T<:CouplingLayer} = CouplingLayer(CouplingAxes(d, mask, n=n); kws...)



