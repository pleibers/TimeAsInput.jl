using ChainRulesCore: @thunk, Tangent, NoTangent, ZeroTangent, HasReverseMode, RuleConfig, rrule_via_ad, unthunk
import ChainRulesCore
using Flux: @functor
using Flux
using LinearAlgebra
using ForwardDiff, ReverseDiff

abstract type AbstractParameterModel end

mutable struct LinearParameterModel{MV<:AbstractVecOrMat} <: AbstractParameterModel
    W::MV
    b::MV
end
@functor LinearParameterModel

function affine(θ::AbstractVecOrMat)
    W = uniform_init(size(θ))
    b = θ
    return LinearParameterModel(W, b)
end

(m::AbstractParameterModel)(x::AbstractFloat) = param_at_T(m, x) :: Union{AbstractVector,  AbstractMatrix}
derivative(m::AbstractParameterModel, time::AbstractMatrix) = [norm(derivative(m, t)) for t in time]
second_derivative(m::AbstractParameterModel, time::AbstractMatrix) = [norm(second_derivative(m, t)) for t in time]



function param_at_T(m::LinearParameterModel, x::AbstractFloat)
    return m.W .* x .+ m.b
end

function derivative(m::LinearParameterModel, time::AbstractFloat)
    return m.W
end

function second_derivative(m::LinearParameterModel, time::AbstractFloat)
    return 0
end

mutable struct ARParameterModel{MV<:AbstractVecOrMat} <: AbstractParameterModel
    a₀::MV
    a₁::MV
    a₂::MV
    a₃::MV
    a₄::MV
    a₅::MV
    a₆::MV
    a₇::MV
    a₈::MV
    a₉::MV
end
@functor ARParameterModel

function ar(θ::AbstractVecOrMat)
    as = [θ for i in 1:10]
    return ARParameterModel(as...)
end

function param_at_T(m::ARParameterModel, x::AbstractFloat)
    return m.a₀ .+ m.a₁ .* x .+ m.a₂ .* x^2 .+ m.a₃ .* x^3 .+ m.a₄ .* x^4 .+ m.a₅ .* x^5 .+ m.a₆ .* x^6 .+ m.a₇ .* x^7 .+ m.a₈ .* x^8 .+ m.a₉ .* x^9
end


function derivative(m::ARParameterModel, x::AbstractFloat)
    return m.a₁ .+ 2 .* m.a₂ .* x .+ 3 .* m.a₃ .* x^2 .+ 4 .* m.a₄ .* x^3 .+ 5 .* m.a₅ .* x^4 .+ 6 .* m.a₆ .* x^5 .+ 7 .* m.a₇ .* x^6 .+ 8 .* m.a₈ .* x^7 .+ 9 .* m.a₉ .* x^8
end
function second_derivative(m::ARParameterModel, x::AbstractFloat)
    return 2 .* m.a₂ .+ 6 .* m.a₃ .* x .+ 12 .* m.a₄ .* x^2 .+ 20 .* m.a₅ .* x^3 .+ 30 .* m.a₆ .* x^4 .+ 42 .* m.a₇ .* x^5 .+ 56 .* m.a₈ .* x^6 .+ 72 .* m.a₉ .* x^7
end


mutable struct MLPParameterModel{NN<:Flux.Chain} <: AbstractParameterModel
    mlp::NN
    shape::Tuple
end
@functor MLPParameterModel (mlp,)

function mlp(θ::AbstractVecOrMat)
    nn = build_mlp(n_output=length(vec(θ)))
    shape = size(θ)
    return MLPParameterModel(nn, shape)
end

function param_at_T(m::MLPParameterModel, x::AbstractFloat)
    v = m.mlp([x])
    PM = reshape(v, m.shape)
    return PM
end


derivative(m::MLPParameterModel, time::AbstractFloat) = reshape(Flux.jacobian(m, time)[1], m.shape)
function second_derivative(m::MLPParameterModel, time::AbstractFloat) 
    f(x) = sum(m.mlp(x))
    hess = ForwardDiff.jacobian(x -> ReverseDiff.gradient(f, x), [time])
    return hess
end




@inbounds function ChainRulesCore.rrule(
    ::typeof(param_at_T),
    PM::LinearParameterModel,
    x::AbstractFloat
)
    y = param_at_T(PM, x)
    function PM_pullback(ΔΩ)
        p̄_at_T = NoTangent()
        P̄M = Tangent{LinearParameterModel}(; W=ΔΩ .* x, b=ΔΩ)
        x̄ = @thunk(sum(PM.W .* ΔΩ))
        return p̄_at_T, P̄M, x̄
    end
    return y, PM_pullback
end

@inbounds function ChainRulesCore.rrule(
    ::typeof(param_at_T),
    PM::ARParameterModel,
    x::AbstractFloat
)
    y = param_at_T(PM, x)
    function PM_pullback(ΔΩ)
        p̄_at_T = NoTangent()
        P̄M = Tangent{ARParameterModel}(; a₀=ΔΩ, a₁=ΔΩ .* x, a₂=ΔΩ .* x^2, a₃=ΔΩ .* x^3, a₄=ΔΩ .* x^4, a₅=ΔΩ .* x^5, a₆=ΔΩ .* x^6, a₇=ΔΩ .* x^7, a₈=ΔΩ .* x^8, a₉=ΔΩ .* x^9)
        x̄ = @thunk(sum((PM.a₁ .+ 2 .* PM.a₂ .* x .+ 3 .* PM.a₃ .* x^2 .+ 4 .* PM.a₄ .* x^3 .+ 5 .* PM.a₅ .* x^4 .+ 6 .* PM.a₆ .* x^5 .+ 7 .* PM.a₇ .* x^6 .+ 8 .* PM.a₈ .* x^7 .+ 9 .* PM.a₉ .* x^8) .* ΔΩ))
        return p̄_at_T, P̄M, x̄
    end
    return y, PM_pullback
end

# @inbounds function ChainRulesCore.rrule(
#     config::RuleConfig{>:HasReverseMode},
#     ::typeof(param_at_T),
#     PM::MLPParameterModel,
#     x::AbstractFloat
# )
#     y = param_at_T(PM, x)
#     mlpx, mlp_pullback = rrule_via_ad(config, PM.mlp, [x])
#     _, reshape_pullback = rrule_via_ad(config, reshape, mlpx, PM.shape)
#     function PM_pullback(ΔΩ)
#         p̄_at_T = NoTangent()
#         P̄M = Tangent{MLPParameterModel}(; mlp=ΔΩ, shape=NoTangent())
#         _, mlpx̄,_ = reshape_pullback(ΔΩ)
#         _, x̄ = mlp_pullback(unthunk(mlpx̄))
#         return p̄_at_T, P̄M, x̄
#     end
#     return y, PM_pullback
# end