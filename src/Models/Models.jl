module Models
using Flux: @functor, relu
using BPTT

include("initialization.jl")

include("nltPLRNN.jl")
export nltPLRNN

include("mlpPLRNN.jl")
export mlpPLRNN

include("multiPLRNN.jl")
export multiPLRNN

include("ptPLRNN.jl")
export ptPLRNN
    
end