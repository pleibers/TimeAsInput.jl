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

function predict_TP(Results_path::String, model_name::String, name::String, data_system::String, TP_location::Int)
    model_path = Results_path * model_name
    args_path = Results_path * "args.json"

    args = convert_to_Float32(JSON.parsefile(args_path))
    model = load_model_(model_path)
    og_data = npzread("data/benchmarks/$data_system.npy")
    time_data = npzread("data/time_data/time_$data_system.npy")
    @assert time_data[1] == -1 "data not normalized"
    ext_in = time_data
    println("generating...")
    pred = generate(model, og_data[1, :], ext_in, size(ext_in, 1))

    pred_BTP = pred[1:TP_location, :]
    pred_ATP = pred[TP_location+1:end, :]
    truth = og_data[TP_location:end, :]

    println("plotting...")
    # plot pred_ATP, truth, pred_BTP in 3d in different colours with labels After TP, Before TP, Truth
    plot3d(pred_BTP[:, 1], pred_BTP[:, 2], pred_BTP[:, 3], label="Before TP", color=:green, legend=:topleft, title="After Tipping Point Predictions")
    plot3d!(pred_ATP[:, 1], pred_ATP[:, 2], pred_ATP[:, 3], label="After TP", color=:red)
    plot3d!(truth[:, 1], truth[:, 2], truth[:, 3], label="Truth", color=:blue, linealpha=0.8)
    savefig("Figures/evaluation/$name.png")
    println("done...")
end

sys = "PaperLorenzBigChange"
predict_TP("Results/external_inputs/Paper_pred/", "model_200.bson", "ahead_pred_paper", sys, TP_loc[sys])

sys = "ShrinkingLorenz"
predict_TP("Results/external_inputs/Shrink_pred/", "model_4800.bson", "ahead_pred_shrink", sys, TP_loc[sys])
