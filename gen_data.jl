using Benchmarks
using NPZ

normalize(v::AbstractVector) = 2 .* (v .- minimum(v))./(maximum(v)-minimum(v)) .-1

function main()
    args = parse_commandline()
    generate_benchmarks(args)

    num_T = args["num_T"]::Int
    ΔT = args["delta_T"]::Float32
    name = args["name"]::String

    t = 0.0:ΔT:(num_T*ΔT)
    t = normalize(collect(t))
    time = reshape(t,length(t),1)

    mkpath("data/time_data")
    npzwrite("data/time_data/time_$name.npy", time)
end

main()
