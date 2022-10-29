module TimeAsInput

include("utils.jl")
export linear, Maybe

include("initialization.jl")

include("nltPLRNN.jl")
export nltPLRNN

include("parsing.jl")
export commandline_parsing, args, parse_transform

include("main_routine.jl")
export main_training_routine

end
