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
        default = 2
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

    combinations = Base.Iterators.product(["affine","mlp","ar"], ["relu", "tanh"])
    opt_model_args = Vector{Vector{String}}()
    for comb in combinations
        x = collect(comb)
        push!(opt_model_args, x)
    end

    # list arguments here
    args = BPTT.ArgVec([
        Argument("path_to_data", ["data/benchmarks/StopBurstBN.npy", "data/benchmarks/PaperLorenzBigChange.npy"], "d"),
        Argument("optional_model_args", opt_model_args, "pm"),
        Argument("hidden_dim", [200,600],"h"),
        Argument("model", ["ptPLRNN","nswPLRNN"],"m"),
        Argument("lat_model_regularization",[0.0,0.1,0.3],"r")
    ])

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks
    pmap(main_training_routine, tasks)
end

ubermain(ub_args["runs"])