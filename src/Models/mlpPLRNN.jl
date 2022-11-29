using BPTT
using Flux: @functor, relu
using Flux

function build_mlp(; n_hidden=32, n_input=1, n_output=3)
    return Chain(
        Dense(n_input => n_hidden, relu, init=uniform_init),
        Dense(n_hidden => n_output,init=uniform_init))
end

mutable struct mlpPLRNN{V<:AbstractVector,M<:AbstractMatrix,NN<:Flux.Chain} <: AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    mlp::Union{NN,Nothing}
end
@functor mlpPLRNN

function mlpPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    mlp = build_mlp(n_input=K, n_output=M)
    return mlpPLRNN(A, W₁, W₂, h₁, h₂, mlp)
end

function mlpPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    return mlpPLRNN(A, W₁, W₂, h₁, h₂, nothing)
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
