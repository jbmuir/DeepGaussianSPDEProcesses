module DeepGaussianSPDEProcesses_Tests

using DeepGaussianSPDEProcesses
using ForwardDiff
using Test
using Random
using Zygote
using LinearAlgebra

@testset "DeepGaussianSPDEProcesses Tests" begin

    @testset "Scalar l tests" begin
        wlen = 501
        Random.seed!(94899109)
        w = randn(wlen)
        l = exp.(randn())
        σ = exp.(randn())
        h = 1.0
        d = 1
        spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)) / h^2, wlen)
        f(x) = sum(spde(x[1],x[2],x[3:end]))
        g(x) = sum(spde_iter(spde, x[1], x[2], x[3:end]))
        x = [l; σ; w]
        d_zygote = gradient(f, x)[1]
        d_fd = ForwardDiff.gradient(f, x)
        d_zygote_cg = gradient(g, x)[1]
        #test cg solution = exlicit solution
        a=spde(l, σ, w)
        b=spde_iter(spde, l, σ, w)
        tol = sqrt(sum(a.^2)/wlen)/1e6 # we can't expect gmres to hit it exactly so have a slightly weaker tolerance
        gtol = sqrt(sum(d_fd.^2)/wlen)/1e6
        @test all(isapprox(a,b, rtol=tol))
        #test gradients of explicit solution - these should be exact
        @test all(isapprox(d_zygote, d_fd))
        #test gradients of zygote cg solution vs explicit fd solution
        @test all(isapprox(d_zygote_cg, d_fd, rtol=gtol))
    end

    @testset "Vector l tests" begin
        wlen = 501
        Random.seed!(94899109)
        w = randn(wlen)
        l = exp.(randn((wlen)))
        σ = exp.(randn())
        h = 1.0
        d = 1
        spde = MaternSPDE(d, h, Tridiagonal(ones(wlen-1), -2*ones(wlen), ones(wlen-1)) / h^2, wlen)
        f(x) = sum(spde(x[1:wlen],x[wlen+1],x[(wlen+2):end]))
        g(x) = sum(spde_iter(spde, x[1:wlen],x[wlen+1],x[(wlen+2):end]))
        x = [l; σ; w]
        d_zygote = gradient(f, x)[1]
        d_fd = ForwardDiff.gradient(f, x)
        d_zygote_cg = gradient(g, x)[1]
        #test cg solution = exlicit solution
        a=spde(l, σ, w)
        b=spde_iter(spde, l, σ, w)
        tol = sqrt(sum(a.^2)/wlen)/1e6 # we can't expect gmres to hit it exactly so have a slightly weaker tolerance
        gtol = sqrt(sum(d_fd.^2)/wlen)/1e6        
        @test all(isapprox(a,b, rtol=tol))
        #test gradients of explicit solution- these should be exact
        @test all(isapprox(d_zygote, d_fd))
        #test gradients of zygote cg solution vs explicit fd solution
        @test all(isapprox(d_zygote_cg, d_fd, rtol=gtol))
    end

end

end
