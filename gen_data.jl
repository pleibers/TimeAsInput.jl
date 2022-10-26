using Benchmarks
using NPZ


function main()
    args = parse_commandline()
    generate_benchmarks(args)

    num_T = args["num_T"]::Int
    ΔT = args["delta_T"]::Float32
    name = args["name"]::String

    t = 0.0:ΔT:(num_T*ΔT)
    time = reshape(t,length(t),1)

    mkpath("data/time_data")
    npzwrite("data/time_data/time_$name.npy", time)
end

main()