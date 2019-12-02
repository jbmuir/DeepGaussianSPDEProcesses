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
    spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)) / h^2, wlen)
    f(x) = sum(spde(x[1],x[2],x[3:end]))
    g(x) = sum(spde_cg(spde, x[1], x[2], x[3:end]))
    x = [l; σ; w]
    d_zygote = gradient(f, x)[1]
    d_fd = ForwardDiff.gradient(f, x)
    d_zygote_cg = gradient(g, x)[1]
    #test cg solution = exlicit solution
    @test all(isapprox(spde(l, σ, w), spde_cg(spde, l, σ, w)))
    #test gradients of explicit solution
    @test all(isapprox(d_zygote, d_fd))
    #test gradients of zygote cg solution vs explicit fd solution
    @test all(isapprox(d_zygote_cg, d_fd))
end

@testset "Vector l tests" begin
    wlen = 501
    Random.seed!(94899109)
    w = randn(wlen)
    l = ones(wlen)
    σ = 1.0
    h = 1.0
    d = 1
    spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)) / h^2, wlen)
    f(x) = sum(spde(x[1:wlen],x[wlen+1],x[(wlen+2):end]))
    g(x) = sum(spde_cg(spde, x[1:wlen],x[wlen+1],x[(wlen+2):end]))
    x = [l; σ; w]
    d_zygote = gradient(f, x)[1]
    d_fd = ForwardDiff.gradient(f, x)
    d_zygote_cg = gradient(g, x)[1]
    #test cg solution = exlicit solution
    @test all(isapprox(spde(l, σ, w), spde_cg(spde, l, σ, w)))
    #test gradients of explicit solution
    @test all(isapprox(d_zygote, d_fd))
    #test gradients of zygote cg solution vs explicit fd solution
    @test all(isapprox(d_zygote_cg, d_fd))
end

end
