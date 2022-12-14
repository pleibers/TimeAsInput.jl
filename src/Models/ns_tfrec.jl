using BPTT
using BPTT.TFTraining: AbstractTFRecur
import BPTT.TFTraining: choose_recur_wrapper

abstract type AbstractNSTFRecur <: AbstractTFRecur end

"""
Inspired by `Flux.Recur` struct, which by default has no way
of incorporating teacher forcing.

This is just a convenience wrapper around stateful models,
to be used during training.
"""
mutable struct nsTFRecur{M<:AbstractMatrix} <: AbstractNSTFRecur
    # stateful model, e.g. PLRNN
    model::Any
    # observation model
    O::ObservationModel
    # state of the model
    z::M
    # forcing interval
    const τ::Int
end
Flux.@functor nsTFRecur

function (tfrec::nsTFRecur)(x::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    z = tfrec.model(z)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

function (tfrec::nsTFRecur)(x::AbstractMatrix, s::AbstractMatrix, t::Int)
    # determine if it is time to force the model
    z = tfrec.z

    # perform one step using the model, update model state
    # precompute the non stationary parameters
    params = @views [get_params_at_T(tfrec.model, s[:, i]) for i in axes(s, 2)]
    zₜ = ns_step.(eachcol(z), params)
    z = reduce(hcat, zₜ)

    # force
    z̃ = (t - 1) % tfrec.τ == 0 ? force(z, x) : z
    tfrec.z = z̃
    return z
end

# Weak TF Recur
mutable struct nsWeakTFRecur{M<:AbstractMatrix} <: AbstractNSTFRecur
    # stateful model, e.g. PLRNN
    model::Any
    # ObservationModel
    O::ObservationModel
    # state of the model
    z::M
    # weak forcing α
    const α::Float32
end
Flux.@functor nsWeakTFRecur

function (tfrec::nsWeakTFRecur)(z⃰::AbstractMatrix, t::Int)
    z = tfrec.z
    D, M = size(z⃰, 1), size(z, 1)
    z = tfrec.model(z)
    # weak tf
    z̃ = @views force(z[1:D, :], z⃰, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end

function (tfrec::nsWeakTFRecur)(z⃰::AbstractMatrix, s::AbstractMatrix, t::Int)
    z = tfrec.z
    D, M = size(z⃰, 1), size(z, 1)
    # precompute the non stationary parameters
    params = @views [get_params_at_T(tfrec.model, s[:, i]) for i in axes(s, 2)]

    zₜ = ns_step.(eachcol(z), params)
    z = reduce(hcat, zₜ)
    # weak tf
    z̃ = @views force(z[1:D, :], z⃰, tfrec.α)
    z̃ = (D == M) ? z̃ : force(z, z̃)

    tfrec.z = z̃
    return z
end


function choose_recur_wrapper(
    m::ptPLRNN,
    d::AbstractDataset,
    O::ObservationModel,
    M::Int,
    N::Int,
    S::Int,
    τ::Int,
    α::Float32,
)
    init_z = similar(d.X, M, S)
    if N ≥ M
        println("N(=$N) ≥ M(=$M), using weak TF with α = $α")
        return nsWeakTFRecur(m, O, init_z, α)
    else
        if α < 1.0f0
            println("N(=$N) < M(=$M) and α < 1 --> using weak TF with α = $α")
            return nsWeakTFRecur(m, O, init_z, α)
        elseif α == 1.0f0
            println("N(=$N) < M(=$M) and α = 1 --> using sparse TF with τ = $τ")
            return nsTFRecur(m, O, init_z, τ)
        end
    end
end