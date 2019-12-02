struct MaternSPDE{T<:Integer,U<:Real,S<:AbstractMatrix}
    d::T #Dimension of SPDE
    h::U #Discretization length
    D::S #Laplacian Differential Operator
    N::Int #Number of cells
end

function (m::MaternSPDE)(l::Real, σ::Real, w::AbstractVector)
    return (I - l^2 * m.D) \ (σ * sqrt(l^m.d) * w / m.h^m.d)
end

@adjoint function (m::MaternSPDE)(l::Real, σ::Real, w::AbstractVector)
    L = (I - l^2 * m.D)
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
    return (I - l^2 * m.D) \ (σ * sqrt(l^m.d) * w / m.h^m.d)
end

@adjoint function (m::MaternSPDE)(l::AbstractVector, σ::Real, w::AbstractVector)
	l = Diagonal(l)
    L = (I - l^2 * m.D)
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