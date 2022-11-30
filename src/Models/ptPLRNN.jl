using BPTT
using Flux: @functor, relu

# shallowPLRNN where parameters are inferred from time
mutable struct ptPLRNN <: BPTT.AbstractShallowPLRNN
    Aₜ::AbstractParameterModel
    W₁ₜ::AbstractParameterModel
    W₂ₜ::AbstractParameterModel
    h₁ₜ::AbstractParameterModel
    h₂ₜ::AbstractParameterModel
end
@functor ptPLRNN

# initialization/constructor
# no options without external inputs, as it is just a shallowPLRNN then
# no options for multidimensional input (yet)
function ptPLRNN(M::Int, hidden_dim::Int, PM_type::String,K::Int)
    @assert K==1 "external input dimension $K != 1, is not supported yet"
    init_PM = @eval $(Symbol(PM_type))
    A, _, h₁ = init_PM.(initialize_A_W_h(M))
    h₂ = init_PM(zeros(Float32, hidden_dim))
    W₁, W₂ = init_PM.(initialize_Ws(M, hidden_dim))
    return ptPLRNN(A, W₁, W₂, h₁, h₂)
end

function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractVector, t::AbstractVector)
    return m.Aₜ(t) .* z .+ m.W₁ₜ(t) * relu.(m.W₂ₜ(t) * z .+ m.h₂ₜ(t)) .+ m.h₁ₜ(t)
end

function BPTT.PLRNNs.step(m::ptPLRNN, z::AbstractMatrix, t::AbstractMatrix)
    zₜ = m.(eachcol(z),eachcol(t))
    return reduce(hcat, zₜ) 
end