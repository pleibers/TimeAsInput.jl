

linear(a::Int,b::Int,t::Int) = a * x +b
linear(A::AbstractVector, B::AbstractVector, t::Int) = A * t + B
