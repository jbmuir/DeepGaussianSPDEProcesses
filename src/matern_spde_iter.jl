function _matern_lhs(l::T, D::S, x::U) where {T<:Real, S<:AbstractMatrix, U<:AbstractVector}
    return x .- l * l * (D * x)
end

function spde_iter(m::MaternSPDE, l::Real, σ::Real, w::AbstractVector{T}) where T
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end   
    L = LinearMap{T}(lhs, lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true) 
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    return gmres(L, σ * s * w)
end

@adjoint function spde_iter(m::MaternSPDE, l::Real, σ::Real, w::AbstractVector{T}) where T
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end   
    L = LinearMap{T}(lhs, lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true)     
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    v = gmres(L, σ * s * w)
    return v, Δ -> begin
    					LtΔ = gmres(L', Δ)
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

function spde_iter(m::MaternSPDE, l::AbstractVector{T}, σ::Real, w::AbstractVector{T}) where T
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end
    L = LinearMap{T}(lhs, lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true)
    ld = l.^m.d
    s = sqrt.(ld) / m.h^m.d
    return gmres(L, σ * s .* w)
end

@adjoint function spde_iter(m::MaternSPDE, l::AbstractVector{T}, σ::Real, w::AbstractVector{T}) where T
    lhs = let l = l, D = m.D
        x -> _matern_lhs(l, D, x)
    end
    L = LinearMap{T}(lhs, lhs, m.N, issymmetric=true, ishermitian=true, isposdef=true)
    ld = l.^m.d
    s = sqrt.(ld) / m.h^m.d
    sw = s .* w
    v = gmres(L, σ * sw)
    return v, Δ -> begin
					LtΔ = gmres(L', Δ)
			    	(nothing, 
			    	((2 * l .* (m.D * v)) + m.d / 2 * σ * (sw ./ l)) .* LtΔ, 
			    	sw' * LtΔ, 
			     	σ * s .* LtΔ)
			    end
end