
using Flux: @functor
using Flux
using LinearAlgebra
using Zygote: hessian

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

(m::AbstractParameterModel)(x::AbstractFloat) = param_at_T(m, x)::Union{AbstractVector,AbstractMatrix}
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
end
@functor ARParameterModel

function ar(θ::AbstractVecOrMat)
    as = (uniform_init(size(θ)) for i in 1:7)
    return ARParameterModel(as...)
end

function param_at_T(m::ARParameterModel, x::AbstractFloat)
    return m.a₀ .+ m.a₁ .* x .+ m.a₂ .* x^2 .+ m.a₃ .* x^3 .+ m.a₄ .* x^4 .+ m.a₅ .* x^5 .+ m.a₆ .* x^6
end


function derivative(m::ARParameterModel, x::AbstractFloat)
    return m.a₁ .+ 2 .* m.a₂ .* x .+ 3 .* m.a₃ .* x^2 .+ 4 .* m.a₄ .* x^3 .+ 5 .* m.a₅ .* x^4 .+ 6 .* m.a₆ .* x^5 
end
function second_derivative(m::ARParameterModel, x::AbstractFloat)
    return 2 .* m.a₂ .+ 6 .* m.a₃ .* x .+ 12 .* m.a₄ .* x^2 .+ 20 .* m.a₅ .* x^3 .+ 30 .* m.a₆ .* x^4
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


derivative(m::MLPParameterModel, time::AbstractFloat) = Flux.gradient(x->sum(m(x)), time)
function second_derivative(m::MLPParameterModel, time::AbstractFloat)
    f(x) = sum(m.mlp([x]))
    return hessian(f, time)
end

mutable struct VARParameterModel{M<:AbstractMatrix, MV<:AbstractVecOrMat} <: AbstractParameterModel
    P₀::MV
    R::M
    B::MV
end
@functor VARParameterModel

function var(θ::AbstractVecOrMat)
    P₀ = θ
    R = Matrix{Float32}(1.0I, size(θ,1), size(θ,1))
    B = zeros(Float32,size(θ))
    return VARParameterModel(P₀, R, B)
end

function param_at_T(m::VARParameterModel, time::AbstractFloat)
    t = Int(time)
    if m.R == I
        Pt = m.R^t * m.P₀ .+ t * m.R^(t - 1) * m.B
    else
        Pt = (I - m.R^t) * inv(I - m.R) * m.B .+ m.R^t * m.P₀
    end
    return Pt
end

function next_param(m::VARParameterModel,pt::AbstractVecOrMat)
    return m.R * pt .+ m.B
end
