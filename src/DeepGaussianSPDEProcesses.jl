module DeepGaussianSPDEProcesses

using LinearAlgebra: Diagonal, I
using Zygote: @adjoint
using LinearMaps
using IterativeSolvers

export MaternSPDE, spde_cg

include("matern_spde.jl")
include("matern_spde_cg.jl")

end
