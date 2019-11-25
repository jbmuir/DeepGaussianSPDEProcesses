module DeepGaussianSPDEProcesses_Tests

using DeepGaussianSPDEProcesses
using ForwardDiff
using Test
using Random
using Zygote
using LinearAlgebra


@testset "Scalar l tests" begin
    wlen = 501
    Random.seed!(94899109)
    w = randn(wlen)
    l = 1.0
    σ = 1.0
    h = 1.0
    d = 1
    spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)) / h^2)
    f(x) = sum(spde(x[1],x[2],x[3:end]))
    x = [l; σ; w]
    d_zygote = gradient(f, x)[1]
    d_fd = ForwardDiff.gradient(f, x)
    @test all(isapprox(d_zygote, d_fd))
end

@testset "Vector l tests" begin
    wlen = 501
    Random.seed!(94899109)
    w = randn(wlen)
    l = ones(wlen)
    σ = 1.0
    h = 1.0
    d = 1
    spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)) / h^2)
    g(x) = sum(spde(x[1:wlen],x[wlen+1],x[(wlen+2):end]))
    x = [l; σ; w]
    d_zygote = gradient(g, x)[1]
    d_fd = ForwardDiff.gradient(g, x)
    @test all(isapprox(d_zygote, d_fd))
end

end
