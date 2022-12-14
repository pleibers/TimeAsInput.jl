using Flux


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

function mlpPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    mlp = build_mlp(n_input=K, n_output=M)
    return mlpPLRNN(A, W₁, W₂, h₁, h₂, nothing, mlp)
end

function mlpPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    return mlpPLRNN(A, W₁, W₂, h₁, h₂, nothing, nothing)
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
