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
# Contains functions related to NICE affine couplings
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export NICECouplingLayer

############
# NICE layer structure

struct NICECouplingLayer{T<:Flux.Chain} <: AffineCouplingLayer
    
    t_net::T  
    axes::AffineCouplingAxes
    
end

@auto_flow NICECouplingLayer [:t_net]