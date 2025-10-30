using DensityFlows
using Test

@testset "data" begin
    
    x = 0.2f0 * ones(Float32, 7, 10) 
    θ = 0.1f0 * ones(Float32, 2, 10)

    x[1, 2] = 0.3f0
    θ[1, 2] = 0.4f0

    metadata = MetaData("", 7, 2, minimum(θ, dims=2)[:, 1], maximum(θ, dims=2)[:, 1])

    @test size(dflt_θ(x))[2:end] == size(x)[2:end]
    
    data = DataArrays(x, θ)

    @test number_dimensions(data) == metadata.d
    @test number_conditions(data) == metadata.n
    
    x_t, θ_t = DensityFlows.normalized_training_data(data, metadata)

    @test maximum(x_t) <= 1
    @test maximum(θ_t) <= 1
    @test minimum(x_t) >= 0
    @test minimum(θ_t) >= 0

end

@testset "axes" begin

    data = DataArrays(ones(Float32, 7, 10), ones(Float32, 2, 10))
    @test CouplingAxes(7, [4, 5, 6, 7], n=2) == CouplingAxes(7, 3, n=2)
    @test CouplingAxes(data) == CouplingAxes(7, 3, n=2)
    @test CouplingAxes(data, [4, 5, 6, 7]) == CouplingAxes(7, 3, n=2)
    @test CouplingAxes(data, 3) == CouplingAxes(7, 3, n=2)
    
end

@testset "real_NVP" begin
    
    z1 = 0.2f0 * ones(Float32, 7, 10)
    θ  = 0.1f0 * ones(Float32, 2, 10)

    layer_1 = CouplingLayer(RNVPCouplingLayer, 7, 3, n=2)

    x, ln_det_jac_1 = forward(layer_1, z1, θ)
    z2, ln_det_jac_2 = backward(layer_1, x, θ)
    
    @test z1 ≈ z2
    @test all((ln_det_jac_1 .+ ln_det_jac_2) .≈ 0f0)

    layer_2 = CouplingLayer(RNVPCouplingLayer, 7, [1, 3, 5, 7], n=2)

    x, ln_det_jac_1 = forward(layer_2, z1, θ)
    z2, ln_det_jac_2 = backward(layer_2, x, θ)
    
    @test z1 ≈ z2
    @test all((ln_det_jac_1 .+ ln_det_jac_2) .≈ 0f0)

end

@testset "chain" begin
    
    layer_1 = CouplingLayer(RNVPCouplingLayer, 7, [1, 3, 5, 7], n=2)
    layer_2 = CouplingLayer(RNVPCouplingLayer, 7, [4, 2, 5, 1, 6], n=2)
    block = CouplingBlock(RNVPCouplingLayer, 7, [4, 2, 5, 1], n=2)

    small_chain = FlowChain(layer_1, layer_2)

    @test length(concatenate(small_chain, block)) == 3
    @test length(concatenate(block, small_chain)) == 3
    @test typeof(small_chain[1]) <: CouplingLayer
    
    x1 = 0.2f0 * ones(Float32, 7, 10)
    θ  = 0.1f0 * ones(Float32, 2, 10)

    x1[:, 2] .= 0.4f0
    θ[1, 2] = 0.4f0

    chain = concatenate((small_chain, FlowChain(block, NormalizationLayer(x1))))

    @test typeof(chain) <: FlowChain
    @test typeof(chain[end]) <: NormalizationElement

    z, ln_det_jac_2 = backward(chain, x1, θ)
    x2, ln_det_jac_1 = forward(chain, z, θ)

    @test x1 ≈ x2
    @test all((ln_det_jac_1 .+ ln_det_jac_2) .≈ 0f0)
    
end