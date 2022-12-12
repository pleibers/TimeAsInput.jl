include("paramModel.jl")

# shallowPLRNN where parameters are inferred from time
mutable struct ptPLRNN <: BPTT.AbstractShallowPLRNN
    Aₜ::AbstractParameterModel
    W₁ₜ::AbstractParameterModel
    W₂ₜ::AbstractParameterModel
    h₁ₜ::AbstractParameterModel
    h₂ₜ::AbstractParameterModel

    t::AbstractMatrix
end
@functor ptPLRNN (Aₜ,W₁ₜ,W₂ₜ,h₁ₜ,h₂ₜ)

# initialization/constructor
# no options without external inputs, as it is just a shallowPLRNN then
# no options for multidimensional input (yet)
function ptPLRNN(M::Int, hidden_dim::Int, PM_type::String, K::Int)
    @assert K == 1 "external input dimension $K != 1, is not supported yet"
    init_PM = @eval $(Symbol(PM_type))
    A, _, h₁ = init_PM.(initialize_A_W_h(M))
    h₂ = init_PM(zeros(Float32, hidden_dim))
    W₁, W₂ = init_PM.(initialize_Ws(M, hidden_dim))
    return ptPLRNN(A, W₁, W₂, h₁, h₂, randn(Float32, 10, 1))
end

function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractVector, time::AbstractVector)
    t = time[1]
    return m.Aₜ(t) .* z .+ m.W₁ₜ(t) * relu.(m.W₂ₜ(t) * z .+ m.h₂ₜ(t)) .+ m.h₁ₜ(t)
end

function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractMatrix, t::AbstractMatrix)
    zₜ = m.(eachcol(z), eachcol(t))
    return reduce(hcat, zₜ)
end

# has a weird way of getting lambda maybe changing it in bptt?
function BPTT.TFTraining.regularize(m::ptPLRNN, λ::Float32; penalty=l2_penalty,λ₂=0.1)
    A_reg_1 = penalty(derivative(m.Aₜ, m.t))
    W₁_reg_1 = penalty(derivative(m.W₁ₜ, m.t))
    W₂_reg_1 = penalty(derivative(m.W₂ₜ, m.t))
    h₁_reg_1 = penalty(derivative(m.h₁ₜ, m.t))
    h₂_reg_1 = penalty(derivative(m.h₂ₜ, m.t))

    A_reg_2 = penalty(second_derivative(m.Aₜ, m.t))
    W₁_reg_2 = penalty(second_derivative(m.W₁ₜ, m.t))
    W₂_reg_2 = penalty(second_derivative(m.W₂ₜ, m.t))
    h₁_reg_2 = penalty(second_derivative(m.h₁ₜ, m.t))
    h₂_reg_2 = penalty(second_derivative(m.h₂ₜ, m.t))
    λ₁=λ
    return λ₁ * (A_reg_1 + W₁_reg_1 + W₂_reg_1 + h₁_reg_1 + h₂_reg_1) + λ₂ * (A_reg_2 + W₁_reg_2 + W₂_reg_2 + h₁_reg_2 + h₂_reg_2)
end
l2_penalty(θ) = isnothing(θ) ? 0 : sum(abs2, θ)
l1_penalty(θ) = isnothing(θ) ? 0 : sum(abs, θ)
