using ChainRulesCore: @thunk, Tangent, NoTangent, ZeroTangent
import ChainRulesCore

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

(m::LinearParameterModel)(x::AbstractVector) = param_at_T(m,x)
function param_at_T(m::LinearParameterModel, x::AbstractVector)
    return m.W .* x .+ m.b
end

@inbounds function ChainRulesCore.rrule(
    ::typeof(param_at_T),
    PM::LinearParameterModel,
    x::AbstractVector
)
    y = param_at_T(PM, x)
    function PM_pullback(ΔΩ)
        p̄_at_T = NoTangent()
        P̄M = Tangent{LinearParameterModel}(; W=ΔΩ .* x, b=ΔΩ)
        x̄ = NoTangent() #@thunk(PM.W .* ΔΩ)
        return p̄_at_T, P̄M, x̄
    end
    return y, PM_pullback
end

using ChainRulesTestUtils
