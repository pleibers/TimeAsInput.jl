module TimeAsInput

include("utils.jl")
export linear

include("parsing.jl")
export commandline_parsing, args

include("main_routine.jl")
export main_training_routine

end
