abstract type AbstractNSPLRNN <: BPTT.AbstractShallowPLRNN end

# shallowPLRNN where parameters are inferred from time
mutable struct ptPLRNN{APM<:AbstractParameterModel,APV<:AbstractParameterModel,M<:AbstractMatrix, f<:Function} <: AbstractNSPLRNN
    Aₜ::APV
    W₁ₜ::APM
    W₂ₜ::APM
    h₁ₜ::APV
    h₂ₜ::APV

    t::M
    Φ::f
end
@functor ptPLRNN (Aₜ, W₁ₜ, W₂ₜ, h₁ₜ, h₂ₜ)

# initialization/constructor
# no options without external inputs, as it is just a shallowPLRNN then
# no options for multidimensional input (yet)
function ptPLRNN(M::Int, hidden_dim::Int, PM_type::String, activation_fun::String, K::Int)
    @assert K == 1 "external input dimension $K != 1, is not supported yet"
    init_PM = @eval $(Symbol(PM_type))
    A, _, h₁ = init_PM.(initialize_A_W_h(M))
    h₂ = init_PM(zeros(Float32, hidden_dim))
    W₁, W₂ = init_PM.(initialize_Ws(M, hidden_dim))
    Φ = @eval $(Symbol(activation_fun))
    return ptPLRNN(A, W₁, W₂, h₁, h₂, randn(Float32, 10, 1), Φ)
end

function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractVecOrMat)
    return m.Aₜ(0) .* z .+ m.W₁ₜ(0) * relu.(m.W₂ₜ(0) * z .+ m.h₂ₜ(0)) .+ m.h₁ₜ(0)
end

function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractVector, time::AbstractVector)
    t = time[1]
    return m.Aₜ(t) .* z .+ m.W₁ₜ(t) * relu.(m.W₂ₜ(t) * z .+ m.h₂ₜ(t)) .+ m.h₁ₜ(t)
end
function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractMatrix, time::AbstractMatrix)
    z = m.(eachcol(z), eachcol(time))
    return reduce(hcat, z)
end

function get_params_at_T(m::ptPLRNN, time::AbstractVector)
    t = time[1]
    return m.Aₜ(t), m.W₁ₜ(t), m.W₂ₜ(t), m.h₁ₜ(t), m.h₂ₜ(t)
end


function BPTT.TFTraining.regularize(m::ptPLRNN, λ::Float32; penalty=l2_penalty)
    A_reg_1 = @views penalty(derivative(m.Aₜ, m.t))
    W₁_reg_1 = @views penalty(derivative(m.W₁ₜ, m.t))
    W₂_reg_1 = @views penalty(derivative(m.W₂ₜ, m.t))
    h₁_reg_1 = @views penalty(derivative(m.h₁ₜ, m.t))
    h₂_reg_1 = @views penalty(derivative(m.h₂ₜ, m.t))

    A_reg_2 = @views penalty(second_derivative(m.Aₜ, m.t))
    W₁_reg_2 = @views penalty(second_derivative(m.W₁ₜ, m.t))
    W₂_reg_2 = @views penalty(second_derivative(m.W₂ₜ, m.t))
    h₁_reg_2 = @views penalty(second_derivative(m.h₁ₜ, m.t))
    h₂_reg_2 = @views penalty(second_derivative(m.h₂ₜ, m.t))
    λ₁ = λ
    return λ₁ * (A_reg_1 + W₁_reg_1 + W₂_reg_1 + h₁_reg_1 + h₂_reg_1 + A_reg_2 + W₁_reg_2 + W₂_reg_2 + h₁_reg_2 + h₂_reg_2)
end
l2_penalty(θ) = isnothing(θ) ? 0 : sum(abs2, θ)
l1_penalty(θ) = isnothing(θ) ? 0 : sum(abs, θ)

