
##################################################################################
# This file is part of DensityFlows.jl
#
# Copyright (c) 2026, Gaétan Facchinetti
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
# Contains functions related to computing the coverage probability
#
# author: Gaetan Facchinetti
# email: gaetanfacc@gmail.com
#
##################################################################################

export coverage_probability

function coverage_probability(
    flow::Flow{T}, 
    x::AbstractMatrix{T},
    x_min::AbstractVector{T},
    x_max::AbstractVector{T}, 
    θ::AbstractMatrix{T} = dflt_θ(x);
    s::Int=100,
    αs::Union{Nothing, AbstractVector{T}} = nothing,
    n_rejection::Int = 100
    ) where {T}

    # α-level credible masses
    if αs === nothing
        # default values evaluated
        αs = T.([0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 
            0.01, 0.025, 0.0455, 0.1, 0.2, 0.3173, 0.4, 0.5, 
            0.6, 0.7, 0.8, 0.9, 0.95])
    end

    # dimensions
    n, mx = size(x) 
    p, mθ = size(θ)
    @assert mx == mθ "Mismatch between x and θ sizes"

    # posterior_samples[i] is (p, s) matrix of posterior samples for test point i
    # this posterior sampled uses rejection sampling to remain within the prior range
    x_samples = Vector{Matrix{T}}(undef, mx)

    # prepare a table of "good" test, the one we can get enough samples from with the rejection
    good_test = []

    # the sample condition is imposed by the prior bounds
    sample_condition(x::AbstractArray{T}, θ::AbstractArray{T}) where {T} = all(x_min .<= x .<= x_max)

    # loop over all runs passed as input
    for j in 1:mθ
        try
            # resample from the condition
            x_samples[j] = sample_with_rejection(sample_condition, flow, s, Tuple(θ[:, j]), n_rejection) 
            push!(good_test, j) 
        catch
            nothing
        end
    end

    # output some info about the discarded samples
    if length(good_test) <= mθ
        
        # evaluate the number of "bad" tests
        n_bad = mθ - length(good_test)

        if n_bad == 1
            @info "1 sample has been discarded, consider increasing n_rejection arguments if necessary."
        else
            @info "$n_bad samples have been discarded, consider increasing n_rejection arguments if necessary."
        end

    end

    # coverage per parameter dimension
    coverage = zeros(T, p, length(αs), length(good_test))

    # average and spread of the coverage
    μα = zeros(T, p, length(αs))
    σα = zeros(T, p, length(αs))


    """
        central_interval_1d(samples::Vector, α)

    Return lower and upper bounds of the central α-credible interval.
    """
    function central_interval_1d(samples::Vector{Float32}, α)
        lower = Statistics.quantile(samples, (1 - α)/2)
        upper = Statistics.quantile(samples, (1 + α)/2)
        return lower, upper
    end

    # main loop over parameters
    # for every vector x
    for j in 1:n 
        
        # loop over all values of α
        for (k, α) in enumerate(αs)

            # loop on each "good" test simulation 
            for i in eachindex(good_test)
                
                # extract posterior samples for parameter j
                sj = x_samples[good_test[i]][j, :]

                # compute credible interval
                lower, upper = central_interval_1d(sj, α)

                # check if truth is inside
                # add to coverage if truth is inside
                (lower <= x[j, good_test[i]] <= upper) && (coverage[j, k, i] = 1)
            end

            # evaluate the mean and dispersion
            μα[j, k] = Statistics.mean(coverage[j, k, :])
            σα[j, k] = sqrt(μα[j, k] * (1-μα[j, k])/mx)

            # we expect that the average of coverage to be equal to the value of α
            # if the coverage is larger we are underconfident
            # if the coverage is smaller we are overconfident
        end
    end

    # we modify the coverage values in terms of quantiles of the normal distribution
    # this is a common definition in the litterature
    zp = [sqrt(2) * SpecialFunctions.erfinv(1-α) for α in αs]
    μp = [sqrt(2) * SpecialFunctions.erfinv(1-μ) for μ in μα]
    μp_min = [sqrt(2) * SpecialFunctions.erfinv(1-μ) for μ in clamp.(μα - 1.96 * σα, 0.0000001, 1.0)] 
    μp_max = [sqrt(2) * SpecialFunctions.erfinv(1-μ) for μ in clamp.(μα + 1.96 * σα, 0.0000001, 1.0)] 

    return zp, μp, μα, μp_min, μp_max
end