module TimeAsInput

include("utils.jl")
export linear

include("initialization.jl")

include("nltPLRNN.jl")
export nltPLRNN

include("mlpPLRNN.jl")
export mlpPLRNN, nlmlpPLRNN, highnlPLRNN

include("parsing.jl")
export commandline_parsing, args_table, parse_transform

include("main_routine.jl")
export main_training_routine

end
