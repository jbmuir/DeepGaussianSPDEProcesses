module DeepGaussianSPDEProcesses

using LinearAlgebra: Diagonal
using Zygote: @adjoint
using LinearMaps
using IterativeSolvers

export MaternSPDE, spde_iter

include("matern_spde.jl")
include("matern_spde_iter.jl")

end
