module DeepGaussianSPDEProcesses

using LinearAlgebra: Diagonal, I
using Zygote: @adjoint

export MaternSPDE

include("matern_spde.jl")

end
