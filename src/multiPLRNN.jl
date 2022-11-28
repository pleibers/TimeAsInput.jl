using BPTT
using Flux: @functor, relu


# shallowPLRNN with t in the non linearity
mutable struct multiPLRNN{V<:AbstractVector,M<:AbstractMatrix} <: BPTT.AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    L::Union{M, Nothing}
    C::Union{M, Nothing}
end
@functor multiPLRNN

# initialization/constructor
function multiPLRNN(M::Int, hidden_dim::Int, N::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    return multiPLRNN(A, W₁, W₂, h₁, h₂, L, nothing)
end

function multiPLRNN(M::Int, hidden_dim::Int, N::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    L = initialize_L(M, N)
    C = uniform_init((M, K))
    return multiPLRNN(A, W₁, W₂, h₁, h₂, L, C)
end

"""
    BPTT.PLRNNs.step(m::nltPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

External Inputs are used inside of the non linearity

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::multiPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)
    return m.A .* z .+ m.C*s .* m.W₁ * relu.(m.W₂ * z .+ m.h₂) .+ m.h₁
end