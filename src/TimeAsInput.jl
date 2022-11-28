module TimeAsInput

include("utils.jl")
export linear, train_test_split

include("initialization.jl")

include("nltPLRNN.jl")
export nltPLRNN

include("mlpPLRNN.jl")
export mlpPLRNN

include("multiPLRNN.jl")
export multiPLRNN

include("parsing.jl")
export commandline_parsing, args_table, parse_transform

include("main_routine.jl")
export main_training_routine

end
