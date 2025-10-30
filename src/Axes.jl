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
# Contains functions related to Axes
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export CouplingAxes, reverse, is_reverse 

struct CouplingAxes
    
    d::Int # total number of dimensions without the conditions
    n::Int # number of conditions
    
    axis_id::Vector{Int}
    axis_af::Vector{Int}
    axis_nn::Vector{Int}
    
end

function summarize(obj::CouplingAxes)
    str_axis_af = length(obj.axis_af) > 0 ? *([string(v) * "," for v in obj.axis_af]...)[1:end-1] : ""
    str_axis_id = length(obj.axis_id) > 0 ? *([string(v) * "," for v in obj.axis_id]...)[1:end-1] : ""
    print("(d,n)=($(obj.d),$(obj.n)); identity=($str_axis_id), transformed=($str_axis_af)")
end


function ==(x::CouplingAxes, y::CouplingAxes)

    (x.d != y.d) && return false
    (x.n != y.n) && return false
    (sort(x.axis_id) != sort(y.axis_id)) && return false
    (sort(x.axis_af) != sort(y.axis_af)) && return false
    (sort(x.axis_nn) != sort(y.axis_nn)) && return false

    return true
    
end


@doc raw"""

    CouplingAxes(d, mask; kws )
    CouplingAxes(d, j=d÷2; kws...)
    CouplingAxes(data, mask)
    CouplingAxes(data, j=d÷2; reverse)
    
Create axes for CouplingLayer.

# Arguments
- `d::Int`: dimension of the flow.
- `j::Int`: dimension cut (default is `d`÷2).
- `mask::AbstractVector{Int}`: dimensions that are affected by the coupling.
- `n::Int`: number of conditions / parameters (default is 0).
- `reverse::Bool`: (default is `false`).

The dimension cut `j` specifies which dimensions are not modified by the layer. 
If `reverse` is `false` the layer acts as the identity on dimensions (1, `j`).
If `reverse` is `true`  the layer acts as the identity on dimensions (`j`+1, `d`).
"""
function CouplingAxes(
    d::Int,
    mask::AbstractVector{Int};
    n::Int = 0
    )

    @assert maximum(mask) <= d "The mask cannot contain values higher than the dimension"

    # dimensions on which we apply Identity
    axis_id = findall(x -> !(x in mask), range(1, d))

    # dimensions on which we apply Affine transformation
    axis_af = mask

    # dimensions passed to the NN, n first for the parameters
    # and then in input gives all the unmodified (Identity) dimensions
    # (because of the triangular nature of the decomposition
    # the affine transformed dimensions only depend on the 
    # unmodified ones)
    axis_nn = vcat(UnitRange(1, n), axis_id .+ n)

    return CouplingAxes(d, n, axis_id, axis_af, axis_nn)

end

function CouplingAxes(
    d::Int,
    j::Int = d ÷ 2;
    n::Int = 0,
    reverse::Bool = false
    )

    mask = !reverse ? UnitRange(j+1, d) : UnitRange(1, j) 
    return CouplingAxes(d, mask, n=n)

end


CouplingAxes(data::DataArrays, mask::AbstractVector{Int}) = CouplingAxes(number_dimensions(data), mask, n = number_conditions(data))
CouplingAxes(data::DataArrays; reverse::Bool = false) = CouplingAxes(number_dimensions(data), number_dimensions(data) ÷ 2, n = number_conditions(data), reverse = reverse)
CouplingAxes(data::DataArrays, j::Int; reverse::Bool = false) = CouplingAxes(number_dimensions(data), j, n = number_conditions(data), reverse = reverse)


@doc raw"""
    
    Base.reverse(axes)

Swap the dimensions that are left unchanged by the layer.
See also [`CouplingAxes`](@ref).
"""
function Base.reverse(axes::CouplingAxes)
    
    # exchange axis_id and axis_af
    axis_nn = vcat(UnitRange(1, axes.n), axes.axis_af .+ axes.n)
    return CouplingAxes(axes.d, axes.n, axes.axis_af, axes.axis_id, axis_nn)

end

function is_reverse(axes_1::CouplingAxes, axes_2::CouplingAxes)
    return all(axes_1.axis_af .== axes_2.axis_id) && all(axes_2.axis_af .== axes_1.axis_id) && (axes_1.n == axes_2.n)
end

