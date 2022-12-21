
mutable struct nswPLRNN{APM<:AbstractParameterModel,V<:AbstractVector,M<:AbstractMatrix,f<:Function} <: AbstractNSPLRNN
    A::V
    W₁ₜ::APM
    W₂ₜ::APM
    h₁::V
    h₂::V

    t::M
    Φ::f
end
@functor nswPLRNN (A, W₁ₜ, W₂ₜ, h₁, h₂)

function nswPLRNN(M::Int, hidden_dim::Int, PM_type::String, activation_fun::String, K::Int)
    @assert K == 1 "external input dimension $K != 1, is not supported yet"
    init_PM = @eval $(Symbol(PM_type))
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = init_PM.(initialize_Ws(M, hidden_dim))
    Φ = @eval $(Symbol(activation_fun))
    return nswPLRNN(A, W₁, W₂, h₁, h₂, randn(Float32, 10, 1), Φ)
end

function BPTT.PLRNNs.step(m::nswPLRNN, z::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ₜ(0) * m.Φ.(m.W₂ₜ(0) * z .+ m.h₂) .+ m.h₁
end

function BPTT.PLRNNs.step(m::nswPLRNN, z::AbstractVector, time::AbstractVector)
    t = time[1]
    return m.A .* z .+ m.W₁ₜ(t) * m.Φ.(m.W₂ₜ(t) * z .+ m.h₂) .+ m.h₁
end

function BPTT.PLRNNs.step(m::nswPLRNN, z::AbstractMatrix, time::AbstractMatrix)
    z = m.(eachcol(z), eachcol(time))
    return reduce(hcat, z)
end
function get_params_at_T(m::nswPLRNN, time::AbstractVector)
    t = time[1]
    return m.A, m.W₁ₜ(t), m.W₂ₜ(t), m.h₁, m.h₂
end

function BPTT.TFTraining.regularize(m::nswPLRNN, λ::Float32; penalty=l2_penalty)
    W₁_reg_1 = @views penalty(derivative(m.W₁ₜ, m.t))
    W₂_reg_1 = @views penalty(derivative(m.W₂ₜ, m.t))

    W₁_reg_2 = @views penalty(second_derivative(m.W₁ₜ, m.t))
    W₂_reg_2 = @views penalty(second_derivative(m.W₂ₜ, m.t))
    λ₁ = λ
    return λ₁ * (W₁_reg_1 + W₂_reg_1 + W₁_reg_2 + W₂_reg_2)
end


mutable struct nsPLRNN{APM<:AbstractParameterModel,V<:AbstractVector,M<:AbstractMatrix,f<:Function} <: AbstractNSPLRNN
    A::V
    W₁::M
    W₂ₜ::APM
    h₁::V
    h₂::V

    t::M
    Φ::f
end
@functor nsPLRNN (A, W₁, W₂ₜ, h₁, h₂)

function nsPLRNN(M::Int, hidden_dim::Int, PM_type::String, activation_fun::String, K::Int)
    @assert K == 1 "external input dimension $K != 1, is not supported yet"
    init_PM = @eval $(Symbol(PM_type))
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    W₂ₜ = init_PM(W₂)
    Φ = @eval $(Symbol(activation_fun))
    return nsPLRNN(A, W₁, W₂ₜ, h₁, h₂, randn(Float32, 10, 1), Φ)
end

function BPTT.PLRNNs.step(m::nsPLRNN, z::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * m.Φ.(m.W₂ₜ(0) * z .+ m.h₂) .+ m.h₁
end

function BPTT.PLRNNs.step(m::nsPLRNN, z::AbstractVector, time::AbstractVector)
    t = time[1]
    return m.A .* z .+ m.W₁ * m.Φ.(m.W₂ₜ(t) * z .+ m.h₂) .+ m.h₁
end

function BPTT.PLRNNs.step(m::nsPLRNN, z::AbstractMatrix, time::AbstractMatrix)
    z = m.(eachcol(z), eachcol(time))
    return reduce(hcat, z)
end
function get_params_at_T(m::nsPLRNN, time::AbstractVector)
    t = time[1]
    return m.A, m.W₁, m.W₂ₜ(t), m.h₁, m.h₂
end

function BPTT.TFTraining.regularize(m::nsPLRNN, λ::Float32; penalty=l2_penalty)
    W₂_reg_1 = @views penalty(derivative(m.W₂ₜ, m.t))

    W₂_reg_2 = @views penalty(second_derivative(m.W₂ₜ, m.t))
    λ₁ = λ
    return λ₁ *  W₂_reg_1  * W₂_reg_2
end