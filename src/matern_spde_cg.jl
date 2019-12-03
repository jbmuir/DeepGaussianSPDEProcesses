function _matern_lhs(l::T, D::S, x::U) where {T<:Real, S<:AbstractMatrix, U<:AbstractVector}
    return x .- l * l * (D * x)
end

function spde_cg(m::MaternSPDE, l::Real, σ::Real, w::AbstractVector)
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end   
    L = LinearMap(lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true) 
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    return cg(L, σ * s * w)
end

@adjoint function spde_cg(m::MaternSPDE, l::Real, σ::Real, w::AbstractVector)
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end   
    L = LinearMap(lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true)     
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    v = cg(L, σ * s * w)
    return v, Δ -> begin
    					LtΔ = cg(L', Δ)
    					sw = s * w
				    	(nothing, 
				    	(2 * l * m.D * v + m.d / 2 * σ / l * sw)' * LtΔ, 
				    	sw' * LtΔ, 
				     	σ * s * LtΔ)
				    end
end

function _matern_lhs(l::T, D::S, x::U) where {T<:AbstractVector, S<:AbstractMatrix, U<:AbstractVector}
    return x .- l .* l .* (D * x)
end

function spde_cg(m::MaternSPDE, l::AbstractVector, σ::Real, w::AbstractVector)
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end
    L = LinearMap(lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true)
    ld = l.^m.d
    s = sqrt.(ld) / m.h^m.d
    return cg(L, σ * s .* w)
end

@adjoint function spde_cg(m::MaternSPDE, l::AbstractVector, σ::Real, w::AbstractVector)
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end
    L = LinearMap(lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true)
    ld = l.^m.d
    s = sqrt.(ld) / m.h^m.d
    sw = s .* w
    v = cg(L, σ * sw)
    return v, Δ -> begin
					LtΔ = cg(L', Δ)
			    	(nothing, 
			    	((2 * l .* (m.D * v)) + m.d / 2 * σ * (sw ./ l)) .* LtΔ, 
			    	sw' * LtΔ, 
			     	σ * s .* LtΔ)
			    end
end