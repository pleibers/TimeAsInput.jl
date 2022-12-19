using TimeAsInput
using BPTT
using Plots
using NPZ: npzread
using Measures
using JSON
using BSON: load
using Flux


function plot3dseries(tseries, title, t; legend=false)
    l_b = legend ? true : nothing
    plot3d(tseries[:,1], tseries[:,2],tseries[:,3], title=title, lc=(:viridis), linez=t, xlabel="X", ylabel="Y", zlabel="Z", label=nothing, legend=l_b)
end

# include("src/TimeAsInput.jl")
# using .TimeAsInput
Base.Float32(V::Vector{Float64}) = Base.Float32.(V)
function convert_to_Float32(dict::Dict)
    for (key, val) in dict
        dict[key] = val isa AbstractFloat ? Float32(val) : val
    end
    return dict
end
load_model_(path::String) = load(path, @__MODULE__)[:model]

function gen_at_t(plrnn::AbstractShallowPLRNN, t::Int, time::AbstractMatrix,og_data)
    time_input = similar(time)
    time_input .= time[t,1]
    # maybe need to adjust load model
    tseries = generate(plrnn, og_data[1,:],time_input, size(time_input,1))
    return tseries
end

function evaluate_snapshots(Results_path::String, model_name::String, name::String, snapshots_path::String)
    model_path = Results_path*model_name
    args_path = Results_path*"args.json"

    args = convert_to_Float32(JSON.parsefile(args_path))
    model = load_model_(model_path)

    og_data = npzread(args["path_to_data"])
    time_data = npzread(args["path_to_inputs"])
    @assert time_data[1] == -1 "data not normalized"

    ext_in = time_data
    println("generating...")
    ts_0 = gen_at_t(model, 1, ext_in,og_data)
    ts_T = gen_at_t(model, size(ext_in, 1), ext_in,og_data)
    ts_a = generate(model, og_data[1,:], ext_in,size(ext_in,1))
    ts = [ts_0, ts_T, ts_a, og_data]

    snap_path = snapshots_path*"snaps_$name"
    snap_0 = npzread(snap_path*"_1.npy")
    snap_T = npzread(snap_path*"_2.npy")
    og_stuff = [snap_0, snap_T, og_data]

    # # calculate klx when we have the proper data sets
    # klx = Vector{AbstractFloat}()
    # scal, stsp_name = decide_on_measure(args["D_stsp_scaling"], args["D_stsp_bins"], size(og_data,2))
    # for idx in axes(og_stuff,1)
    #     D_stsp = state_space_distance(og_stuff[i], ts[i], scal)
    #     push!(klx, round(D_stsp, digits=5))
    # end

    # Txt = ["klx = $(round(klx[i],digits=5))" for i in axes(klx,1)]
    # push!(Txt, "")
    println("plotting...")
    ps = (plot3d(ts[i][1:end, 1], ts[i][1:end, 2], ts[i][1:end, 3], 
            grid=:show, lc=cgrad(:viridis), line_z=1:1:size(ext_in,1), legend=nothing) 
            #,annotations=((0.0,1.25), Txt[i])) 
            for i in axes(ts, 1))

    h2 = scatter([0, 0], [0, 1], line_z=1:1:size(ext_in,1),
        xlims=(1, 1.1), label=nothing, c=:viridis, colorbar_title="\ntime", framestyle=:none)

    l = @layout [grid(2, 2) a{0.035w}]
    p = plot(ps..., h2, layout=l,
        title=["\nseries at t=0" "\nseries at t=T" "\ngenerated series" "\noriginal data" ""],
        plot_title="Stationary series at different t",
        right_margin=3.0Plots.mm)
    mkpath("Figures/evaluation/")
    savefig(p, "Figures/evaluation/$name _snapshots.png")
    println("done!")
    return nothing
end



function compare_nlt_shallow(path::String)
    args = convert_to_Float32(JSON.parsefile(path * "compare1/args.json"))

    og_data = npzread(args["path_to_data"])
    time_data = npzread(args["path_to_inputs"])
    @assert time_data[1] == -1 "data not normalized"
    ext_in = time_data
    @show size(ext_in)

    tss_nlt = []
    tss_shallow = []
    klx = Vector{AbstractFloat}()
    plots = []
    for run in 1:2
        nlt = load_model_(path * "compare$run/nlt.bson")
        shallow = load_model_(path * "compare$run/shallow.bson")


        ts_nlt = generate(nlt, og_data[1, :], ext_in, 15000)
        ts_shallow = generate(shallow, og_data[1, :], ext_in, 15000)

        push!(tss_nlt, ts_nlt)
        push!(tss_shallow)
        # get klx
        ts = [ts_nlt, ts_shallow]
        scal, stsp_name = decide_on_measure(args["D_stsp_scaling"], args["D_stsp_bins"], size(og_data, 2))
        for idx in 1:2
            D_stsp = state_space_distance(og_data, ts[idx], scal)
            push!(klx, round(D_stsp, digits=5))
        end

        nlt_p = plot3d(og_data[:, 1], og_data[:, 2], og_data[:, 3], lc=:blue, margin=0.01Plots.mm)
        plot3d!(nlt_p, ts_nlt[:, 1], ts_nlt[:, 2], ts_nlt[:, 3], grid=true,
            xlabel="x", ylabel="y", zlabel="z",
            label=nothing, lc=:orange, legend=nothing, margin=0.01Plots.mm)
        push!(plots, nlt_p)
        shallow_p = plot3d(og_data[:, 1], og_data[:, 2], og_data[:, 3], lc=:blue, margin=0.01Plots.mm)
        plot3d!(shallow_p, ts_shallow[:, 1], ts_shallow[:, 2], ts_shallow[:, 3], grid=true,
            xlabel="x", ylabel="y", zlabel="z",
            label=nothing, lc=:orange, legend=nothing, margin=0.01Plots.mm)
        push!(plots, shallow_p)
    end
    comp_p = plot(plots[1], plots[2], plots[3], plots[4], margin=0.0Plots.mm, layout=grid(2, 2),
        titlefont=font(8), title=["\n" * " "^70 * "Inputs\nInside\nklx=$(klx[1])" "\n\nOutside\nklx=$(klx[2])" "\nklx=$(klx[3])" "\nklx=$(klx[4])"], plot_titlefont=font(14), plot_title="Comparison between Models")
    mkpath("Figures/evaluation/")
    savefig(comp_p, "Figures/evaluation/Compare_nlt_shallow_$run.png")
end

function check_around_snapshots(Results_path::String, model_name::String)
    model_path = Results_path * model_name
    args_path = Results_path * "args.json"

    args = convert_to_Float32(JSON.parsefile(args_path))
    model = load_model_(model_path)

    og_data = npzread(args["path_to_data"])
    time_data = npzread(args["path_to_inputs"])
    @assert time_data[1] == -1 "data not normalized"
    ext_in = time_data
    ts_a = generate(model, og_data[1, :], ext_in, size(ext_in, 1))
    # display(plot3dseries(ts_a, "all", ext_in))    
    for i in [1, 10, 100, 1000, 8000]
        ts_0 = gen_at_t(model, i, ext_in, og_data)
        ts_T = gen_at_t(model, size(ext_in, 1) - i, ext_in, og_data)
        title0 = "t=$i"
        titleT = "t=$(size(ext_in, 1)-i)"
        p_0 = plot3dseries(ts_0, nothing, ext_in)
        p_T = plot3dseries(ts_T, nothing, ext_in)
        p_c = plot(p_0, p_T, title=[title0 titleT], legend=nothing)
        savefig(p_c, "snap_$i.png")
    end
end

function eval_wo_ext(model_path::String, name::String, fig::String)
    model = load_model_(model_path)
    og_data = npzread(args["path_to_data"])
    time_data = npzread(args["path_to_inputs"])
    time = zeros(size(time_data))
    ts = Matrix(gen_at_t(model, 1, time, og_data))
    p = plot3d(ts[:, 1], ts[:, 2], ts[:, 3], label="0.5")
    for i in 0:10
        time = time .- 1 .+ i * 0.2
        ts = Matrix(gen_at_t(model, 1, time, og_data))
        plot3d!(ts[:, 1], ts[:, 2], ts[:, 3], label="$(round((i*0.2)/2,digits=1))")
    end
    savefig(p, "Figures/evaluation/$fig _$name.png")
end
    
# -------------------------------------------------------------------------------------------
# Evaluations

# compare_nlt_shallow("Results/external_inputs/Compare_nlt_shallow/")


# evaluate_snapshots("Results/external_inputs/ExplodingLorenz_nlt/001/","last_model.bson", "nltPLRNN_exploding", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/ShiftingLorenz_nlt/002/","last_model.bson", "nltPLRNN_Shifting","data/snapshots/")

# evaluate_snapshots("Results/external_inputs/StopBurstBN_nlt/","last_model.bson", "nltPLRNN_SBBN", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/StopBurstBN_nlt/","4050.bson", "nltPLRNN_SBBN_2", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/StopBurstBN_mlp/","last_model.bson", "mlpPLRNN_SBBN", "data/snapshots/")

# evaluate_snapshots("Results/external_inputs/ShrinkingLorenz_nlt/","last_model.bson", "nltPLRNN_ShrinkingLorenz", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/ShrinkingLorenz_nlt/","pre_last.bson", "nltPLRNN_ShrinkingLorenz_2", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/ShrinkingLorenz_nlt/","preprelast.bson", "nltPLRNN_ShrinkingLorenz_3", "data/snapshots/")

evaluate_snapshots("Results/external_inputs/Paper_base/","model_5000.bson", "nlt_Paperlorenz", "data/snapshots/")


# evaluate_snapshots("Results/external_inputs/StopBurstBN_nlt_p1/","last_1.bson", "nltPLRNN_SBBN_p1", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/StopBurstBN_nlt_p1/","last_2.bson", "nltPLRNN_SBBN_p1_2", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/StopBurstBN_mlp_p1/","last_model.bson", "mlpPLRNN_SBBN_p1", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/StopBurstBN_nlt_unc/", "last_model.bson", "nltPLRNN_SBBN_unc_1", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/StopBurstBN_nlt_unc/", "2850.bson", "nltPLRNN_SBBN_unc_2", "data/snapshots/")


# check_around_snapshots("Results/external_inputs/ShiftingLorenz_nlt/002/", "last_model.bson")
# check_around_snapshots("Results/external_inputs/ExplodingLorenz_nlt/001/","last_model.bson")
# check_around_snapshots("Results/external_inputs/StopBurstBN_nlt/", "4050.bson")
# check_around_snapshots("Results/external_inputs/ShrinkingLorenz_nlt/","last_model.bson")

# eval_wo_ext("Results/external_inputs/StopBurstBN_nlt/last_model.bson" , "snapshots_bad")
# eval_wo_ext("Results/external_inputs/StopBurstBN_nlt/4050.bson",  "snapshots_good")
# eval_wo_ext("Results/external_inputs/ShrinkingLorenz_nlt/last_model.bson","snapshots")


# eval_wo_ext("Results/external_inputs/StopBurstBN_nlt_p1/last_1.bson", "StopBurstBN", "snapss")
# eval_wo_ext("Results/external_inputs/StopBurstBN_nlt_p1/last_2.bson", "StopBurstBN", "snapss_nlt_p1")
# eval_wo_ext("Results/external_inputs/StopBurstBN_mlp_p1/last_model.bson", "StopBurstBN", "snapss_mlp_p1")
# eval_wo_ext("Results/external_inputs/StopBurstBN_nlt_unc/last_model.bson","StopBurstBN", "snapss_nlt_unc")
# eval_wo_ext("Results/external_inputs/StopBurstBN_nlt_unc/2850.bson", "StopBurstBN", "snapss")

# generate trajectory at t=5 and make a 3d plot
