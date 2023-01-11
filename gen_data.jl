using Benchmarks
using NPZ

normalize(v::AbstractVector) = 2 .* (v .- minimum(v))./(maximum(v)-minimum(v)) .-1

function main()
    args = Benchmarks.parse_commandline()
    generate_benchmarks(args)

    T = args["T"]::Int
    ΔT = args["delta_T"]::Float32
    name = args["name"]::String

    if occursin("Paper", args["name"])
        T = 800
    end
    
    t = 0.0:ΔT:T
    time = normalize(collect(t))
    time = reshape(time,length(time),1)
    if !isempty(args["exp_name"])
        name=args["exp_name"]
    end
    mkpath("data/time_data")
    npzwrite("data/time_data/time_$name.npy", time)
    npzwrite("data/time_data/realtime_$name.npy", t)
end

# main()
dir = "data/time_data/"
times = readdir(dir)
times = filter(x->!contains(x,"realtime_"),times)
times = filter(x -> !contains(x, "tp_"), times)

for set in times
    data = npzread(dir*set)
    len = size(data,1)
    tp = 1:len
    set = replace(set,"time_"=>"")
    tp = reshape(tp,size(data))
    npzwrite(dir*"tp_"*set, tp)
end

