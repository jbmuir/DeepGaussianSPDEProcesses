struct MaternSPDE{T<:Integer, U<:Real, S<:AbstractMatrix, V<:AbstractMatrix}
    d::T #Dimension of SPDE
    N::Int #Number of cells
    h::U #Discretization length
    D::S #Laplacian Differential Operator
    I::V #Appropriate identity matrix on whatever device we are on (note that using uniform scaling operator sucks on the gpu)
end

function (m::MaternSPDE)(l::Real, σ::Real, w::AbstractVector)
    return (m.I - l^2 * m.D) \ (σ * sqrt(l^m.d) * w / m.h^m.d)
end

@adjoint function (m::MaternSPDE)(l::Real, σ::Real, w::AbstractVector)
    L = (m.I - l^2 * m.D)
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    v = L \ (σ * s * w)
    return v, Δ -> begin
    					LtΔ = L' \ Δ
    					sw = s * w
				    	(nothing, 
				    	(2 * l * m.D * v + m.d / 2 * σ / l * sw)' * LtΔ, 
				    	sw' * LtΔ, 
				     	σ * s * LtΔ)
				    end
end

function (m::MaternSPDE)(l::AbstractVector, σ::Real, w::AbstractVector)
    l = Diagonal(l)
    return (m.I - l^2 * m.D) \ (σ * sqrt(l^m.d) * w / m.h^m.d)
end

@adjoint function (m::MaternSPDE)(l::AbstractVector, σ::Real, w::AbstractVector)
	l = Diagonal(l)
    L = (m.I - l^2 * m.D)
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    v = L \ (σ * s * w)
    return v, Δ -> begin
					LtΔ = L' \ Δ
					sw = s * w
			    	(nothing, 
			    	(2 * l * Diagonal(m.D * v) + m.d / 2 * σ * Diagonal(sw) / l) * LtΔ, 
			    	sw' * LtΔ, 
			     	σ * s * LtΔ)
			    end
end