using Flux: @functor

abstract type AbstractParameterModel end

mutable struct LinearParameterModel <: AbstractParameterModel
    W::Matrix{Float32}
    b::Vector{Float32}
end
@functor LinearParameterModel

function affine(θ::AbstractVecOrMat)
    W = θ
    b = zeros(Float32, size(W, 1))
    return LinearParameterModel(W, b)
end

(m::LinearParameterModel)(x) = m.W .* x .+ m.b