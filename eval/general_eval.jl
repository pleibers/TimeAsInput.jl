using TimeAsInput
using BPTT
using Plots
using NPZ: npzread
using Measures
using JSON
using BSON: load
using Flux
using Benchmarks: TP_loc


Base.Float32(V::Vector{Float64}) = Base.Float32.(V)
function convert_to_Float32(dict::Dict)
    for (key, val) in dict
        dict[key] = val isa AbstractFloat ? Float32(val) : val
    end
    return dict
end
load_model_(path::String) = load(path, @__MODULE__)[:model]

function plot_reconstruction(X̃::AbstractMatrix, X::AbstractMatrix, name::String)
    plot3d(X[:, 1], X[:, 2], X[:, 3], label="Truth", color=:blue,legend=:topleft, title="Reconstruction")
    plot3d!(X̃[:, 1], X̃[:, 2], X̃[:, 3], label="Reconstruction", color=:red, linealpha=0.8)
    savefig("Figures/evaluation/$name.png")
    println("done...")
end

function evaluate(Results_path::String, model_name::String, name::String)
    model_path = Results_path * model_name
    args_path = Results_path * "args.json"

    args = convert_to_Float32(JSON.parsefile(args_path))
    model = load_model_(model_path)[1]
    og_data = npzread(args["path_to_data"])
    time_data = npzread(args["path_to_inputs"])
    @assert time_data[1] == -1 "data not normalized"

    reconstruction = generate(model, og_data[1, :], time_data, size(time_data, 1))
    plot_reconstruction(reconstruction, og_data, name)
end

for i in [20,40,300,700,1500,2000,3000]#,4740,5000]
    evaluate("Results/external_inputs/mydiff/", "checkpoints/model_$i.bson", "reconstruction$(i)_my")
end

