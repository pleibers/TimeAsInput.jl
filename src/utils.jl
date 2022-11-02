
linear(a::Real,b::Real,t::Real) = a * t +b
linear(A::AbstractVector, B::AbstractVector, t::Real) = A * t + B

