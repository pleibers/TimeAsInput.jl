using Distributed
using ArgParse

function parse_ubermain()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--procs", "-p"
        help = "Number of parallel processes/workers to spawn."
        arg_type = Int
        default = 1

        "--runs", "-r"
        help = "Number of runs per experiment setting."
        arg_type = Int
        default = 3
    end
    return parse_args(s)
end

# parse number of procs, number of runs
ub_args = parse_ubermain()

# start workers in BPTT env
addprocs(
    ub_args["procs"];
    exeflags = `--threads=$(Threads.nthreads()) --project=$(Base.active_project())`,
)

# make pkgs available in all processes
@everywhere using BPTT
@everywhere using TimeAsInput
@everywhere ENV["GKSwstype"] = "nul"

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int)
    # load defaults with correct data types
    defaults = parse_args([], args_table())

    # list arguments here
    args = BPTT.ArgVec([
        Argument("model", ["shallowPLRNN", "nltPLRNN"], "model"),
        Argument("weak_tf_alpha", [0.1, 0.05,0.15,0.3],"α"),
        Argument("hidden_dim", [20, 50,100], "H"),
        Argument("affine_transform_coeff", [[10.0,1.0],[0.1,1.0],[0.01,1.0],[[0.1,0.1,0.1],[1.0,1.0,1.0]],[[0.1,0.1,0.1],[0.5,0.5,0.5]],[-0.1,0.1]],"tc")
    ])

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks
    pmap(main_training_routine, tasks)
end

ubermain(ub_args["runs"])