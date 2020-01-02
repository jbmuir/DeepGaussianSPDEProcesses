function _matern_fwd(l2::T, D::S, x::U) where {T<:Real, S<:AbstractMatrix, U<:AbstractVector}
    return x .- l2 * (D * x)
end

_matern_adj(l2::T, D::S, x::U) where {T<:Real, S<:AbstractMatrix, U<:AbstractVector} = _matern_fwd(l2, D', x)

function spde_iter(m::MaternSPDE, l::Real, σ::Real, w::AbstractVector{T}) where T
    l2 = l^2
    fwd = let l2 = l2, D = m.D
        x -> _matern_fwd(l2, D, x)
    end   
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    return linsolve(fwd, σ * s * w)[1]
end

@adjoint function spde_iter(m::MaternSPDE, l::Real, σ::Real, w::AbstractVector{T}) where T
    l2 = l^2
    fwd = let l2 = l2, D = m.D
        x -> _matern_fwd(l2, D, x)
    end   
    adj = let l2 = l2, D = m.D
        x -> _matern_adj(l2, D, x)
    end   
    ld = l^m.d
    s = sqrt(ld) / m.h^m.d
    v = linsolve(fwd, σ * s * w)[1]
    return v, Δ -> begin
    					LtΔ = linsolve(adj, Δ)[1]
    					sw = s * w
				    	(nothing, 
				    	(2 * l * m.D * v + m.d / 2 * σ / l * sw)' * LtΔ, 
				    	sw' * LtΔ, 
				     	σ * s * LtΔ)
				    end
end

function _matern_fwd(l2::T, D::S, x::U) where {T<:AbstractVector, S<:AbstractMatrix, U<:AbstractVector}
    return x .- l2 .* (D * x)
end

function _matern_adj(l2::T, D::S, x::U) where {T<:AbstractVector, S<:AbstractMatrix, U<:AbstractVector}
    return x .- D' * (l2 .* x)
end

function spde_iter(m::MaternSPDE, l::AbstractVector{T}, σ::Real, w::AbstractVector{T}) where T
    l2 = l.*l
    fwd = let l2 = l2, D = m.D
        x -> _matern_fwd(l2, D, x)
    end     
    ld = l.^m.d
    s = sqrt.(ld) / m.h^m.d
    return linsolve(fwd, σ * s .* w)[1]
end

@adjoint function spde_iter(m::MaternSPDE, l::AbstractVector{T}, σ::Real, w::AbstractVector{T}) where T
    l2 = l.*l
    fwd = let l2 = l2, D = m.D
        x -> _matern_fwd(l2, D, x)
    end   
    adj = let l2 = l2, D = m.D
        x -> _matern_adj(l2, D, x)
    end    
    ld = l.^m.d
    s = sqrt.(ld) / m.h^m.d
    sw = s .* w
    v = linsolve(fwd, σ * sw)[1]
    return v, Δ -> begin
					LtΔ = linsolve(adj, Δ)[1]
			    	(nothing, 
			    	((2 * l .* (m.D * v)) + m.d / 2 * σ * (sw ./ l)) .* LtΔ, 
			    	sw' * LtΔ, 
			     	σ * s .* LtΔ)
			    end
end