using BPTT
using Flux: @functor, relu

# shallowPLRNN with t in the non linearity
mutable struct nltPLRNN{V<:AbstractVector,M<:AbstractMatrix} <: BPTT.AbstractShallowPLRNN
    A::V
    W₁::M
    W₂::M
    h₁::V
    h₂::V
    C::Union{M, Nothing}
end
@functor nltPLRNN

# initialization/constructor
function nltPLRNN(M::Int, hidden_dim::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    return nltPLRNN(A, W₁, W₂, h₁, h₂, nothing)
end

function nltPLRNN(M::Int, hidden_dim::Int, K::Int)
    A, _, h₁ = initialize_A_W_h(M)
    h₂ = zeros(Float32, hidden_dim)
    W₁, W₂ = initialize_Ws(M, hidden_dim)
    C = uniform_init((hidden_dim, K))
    return nltPLRNN(A, W₁, W₂, h₁, h₂, C)
end

"""
    BPTT.PLRNNs.step(m::nltPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)

Evolve `z` in time for one step according to the model `m` (equation).

External Inputs are used inside of the non linearity

`z` is either a `M` dimensional column vector or a `M x S` matrix, where `S` is
the batch dimension.

"""
function BPTT.PLRNNs.step(m::nltPLRNN, z::AbstractVecOrMat, s::AbstractVecOrMat)
    return m.A .* z .+ m.W₁ * relu.(m.W₂ * z .+ m.h₂ .+ m.C * s) .+ m.h₁
end