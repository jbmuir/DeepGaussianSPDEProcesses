struct MaternSPDE{T<:Integer,U<:Real,S<:AbstractMatrix}
    d::T #Dimension of SPDE
    h::U #Discretization length
    D::S #Laplacian Differential Operator
end

function (m::MaternSPDE)(l::Real, σ::Real, w::AbstractVector)
    return (I - l^2 * m.D / m.h^2) \ (σ * sqrt(l^m.d) * w / m.h^m.d)
end

@adjoint function (m::MaternSPDE)(l::Real, σ::Real, w::AbstractVector)
    L = (I - l^2 * m.D / m.h^2)
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    dvdσ = L \ (s * w)
    v = σ * dvdσ
    dvdl = L \ (2 * l * m.D * v / m.h^2 + m.d / 2 * s * σ / l * w)
    return v, Δ -> (nothing, dvdl' * Δ, dvdσ' * Δ, L' \ Δ * σ * s)
end

function (m::MaternSPDE)(λ::AbstractVector, σ::Real, w::AbstractVector)
    l = Diagonal(l)
    return (I - l^2 * m.D / m.h^2) \ (σ * sqrt(l^m.d) * w / m.h^m.d)
end
