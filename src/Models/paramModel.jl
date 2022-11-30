using Flux: @functor

abstract type AbstractParameterModel end

mutable struct LinearParameterModel{MV<:AbstractVecOrMat} <: AbstractParameterModel
    W::MV
    b::MV
end
@functor LinearParameterModel

function affine(θ::AbstractVecOrMat)
    W = zeros(Float32, size(θ))
    b = θ
    return LinearParameterModel(W, b)
end

(m::LinearParameterModel)(x) = m.W .* x .+ m.b