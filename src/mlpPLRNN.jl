using BPTT
using Flux: @functor, relu
using Flux

function build_mlp(; n_hidden=32, n_input=1, n_output=3)
    return Chain(
        Dense(n_input => n_hidden, relu),
        Dense(n_hidden => n_output))
end

mutable struct mlpPLRNN{V<:AbstractVector,M<:AbstractMatrix,NN<:Flux.Chain} <: AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    L::Union{M,Nothing}
    mlp::Union{NN,Nothing}
end
@functor mlpPLRNN

function mlpPLRNN(M::Int, hidden_dim::Int, N::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    mlp = build_mlp(n_input=K, n_output=M)
    return mlpPLRNN(A, W₁, W₂, h₁, h₂, L, mlp)
end

function mlpPLRNN(M::Int, hidden_dim::Int, N::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    return mlpPLRNN(A, W₁, W₂, h₁, h₂, L, nothing)
end

mutable struct nlmlpPLRNN{V<:AbstractVector,M<:AbstractMatrix,NN<:Flux.Chain} <: AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    L::Union{M,Nothing}
    mlp::Union{NN,Nothing}
end
@functor nlmlpPLRNN

function nlmlpPLRNN(M::Int, hidden_dim::Int, N::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    mlp = build_mlp(n_input=K, n_output=hidden_dim)
    return nlmlpPLRNN(A, W₁, W₂, h₁, h₂, L, mlp)
end

function nlmlpPLRNN(M::Int, hidden_dim::Int, N::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    return nlmlpPLRNN(A, W₁, W₂, h₁, h₂, L, nothing)
end


# shallowPLRNN with t in the non linearity
mutable struct highnlPLRNN{V<:AbstractVector,M<:AbstractMatrix} <: BPTT.AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    L::Union{M,Nothing}
    C₁::Union{M,Nothing}
    C₂::Union{M,Nothing}
end
@functor highnlPLRNN

# initialization/constructor
function highnlPLRNN(M::Int, hidden_dim::Int, N::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    return highnlPLRNN(A, W₁, W₂, h₁, h₂, L, nothing, nothing)
end

function highnlPLRNN(M::Int, hidden_dim::Int, N::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    C₁, C₂ = initialize_Cs(M,50,1)
    return highnlPLRNN(A, W₁, W₂, h₁, h₂, L, C₁,C₂)
end

"""
    BPTT.PLRNNs.step(m::mlpPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

time is given to a MLP

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::mlpPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * relu.(m.W₂ * z .+ m.h₂) .+ m.h₁ .+ m.mlp(s)
end

function BPTT.PLRNNs.step(m::nlmlpPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * relu.(m.W₂ * z .+ m.h₂ .+ m.mlp(s)) .+ m.h₁
end

function BPTT.PLRNNs.step(m::highnlPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * relu.(m.W₂ * z .+ m.h₂) .+ m.h₁ .+ m.C₁*relu.(m.C₂*s)
end

