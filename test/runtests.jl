module DeepGaussianSPDEProcesses_Tests

using DeepGaussianSPDEProcesses
using FiniteDifferences
using Test
using Random
using Zygote
using LinearAlgebra


@testset "Scalar l tests" begin
    tol = 1e-10
    wlen = 501
    Random.seed!(94899109)
    w = randn(wlen)
    l = 1.0
    σ = 1.0
    h = 1.0
    d = 1
    spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)))
    f(x) = sum(spde(x[1],x[2],x[3:end]))
    x = [l; σ; w]
    d_zygote = gradient(f, x)[1]
    fdm = central_fdm(5,1)
    d_fdm = grad(fdm, f, x)
    @test all(isapprox(d_zygote, d_fdm, rtol=tol))
end

end
