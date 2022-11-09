using TimeAsInput
using BPTT
using Plots
using NPZ: npzread
using Measures
using JSON
using BSON: load

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

function gen_at_t(plrnn::AbstractShallowPLRNN, t::Int, time::AbstractMatrix)
    time_input = similar(time)
    time_input .= time[t,1]
    _, tseries = generate(plrnn, [0.5, 0.5, 0.5],time_input, size(time_input,1))
    return tseries
end

function evaluate_snapshots(Results_path::String, model_name::String, name::String, data_system::String, snapshots_path::String)
    model_path = Results_path*model_name
    args_path = Results_path*"args.json"

    args = convert_to_Float32(JSON.parsefile(args_path))
    model = load_model_(model_path)

    og_data = npzread("data/benchmarks/$data_system.npy")
    time_data = npzread("data/time_data/time_$data_system.npy")
    @assert time_data[1] == -1 "data not normalized"

    trans_coeff = args["affine_transform_coeff"]
    affine_transformation = parse_transform(args["affine_transformation"], trans_coeff)

    external_inputs = Float32.(affine_transformation.(time_data)) # to have type consistency
    ext_in = permutedims(reduce(hcat, external_inputs), (2, 1))

    ts_0 = gen_at_t(model, 1, ext_in)
    ts_T = gen_at_t(model, size(ext_in, 1), ext_in)
    _, ts_a = generate(model, [0.5, 0.5, 0.5], ext_in,15000)

    ts = [ts_0, ts_T, ts_a, og_data]

    snap_path = snapshots_path*"snaps_$data_system"
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
    ps = (plot3d(ts[i][1:end, 1], ts[i][1:end, 2], ts[i][1:end, 3], 
            grid=:show, lc=cgrad(:viridis), line_z=1:1:15000, legend=nothing) 
            #,annotations=((0.0,1.25), Txt[i])) 
            for i in axes(ts, 1))

    h2 = scatter([0, 0], [0, 1], line_z=1:1:15000,
        xlims=(1, 1.1), label=nothing, c=:viridis, colorbar_title="\ntime", framestyle=:none)

    l = @layout [grid(2, 2) a{0.035w}]
    p = plot(ps..., h2, layout=l,
        title=["\nseries at t=0" "\nseries at t=T" "\ngenerated series" "\noriginal data" ""],
        plot_title="Stationary series at different t",
        right_margin=3.0Plots.mm)
    mkpath("Figures/evaluation/")
    savefig(p, "Figures/evaluation/$name _snapshots.png")
    return nothing
end



function compare_nlt_shallow(path::String, data_system::String)
    args = convert_to_Float32(JSON.parsefile(path*"compare1/args.json"))

    og_data = npzread("data/benchmarks/$data_system.npy")
    time_data = npzread("data/time_data/time_$data_system.npy")
    @assert time_data[1] == -1 "data not normalized"

    trans_coeff = args["affine_transform_coeff"]
    affine_transformation = parse_transform(args["affine_transformation"], trans_coeff)

    external_inputs = Float32.(affine_transformation.(time_data)) # to have type consistency
    ext_in = permutedims(reduce(hcat, external_inputs), (2, 1))

    tss_nlt = []
    tss_shallow = []
    klx = Vector{AbstractFloat}()
    plots = []
    for run in 1:2
        nlt = load_model_(path*"compare$run/nlt.bson")
        shallow = load_model_(path*"compare$run/shallow.bson")


        _, ts_nlt = generate(nlt, [0.5, 0.5, 0.5], ext_in,15000)
        _, ts_shallow = generate(shallow, [0.5,0.5,0.5], ext_in, 15000)

        push!(tss_nlt, ts_nlt)
        push!(tss_shallow)
        # get klx
        ts = [ts_nlt, ts_shallow]
        scal, stsp_name = decide_on_measure(args["D_stsp_scaling"], args["D_stsp_bins"], size(og_data,2))
        for idx in 1:2
            D_stsp = state_space_distance(og_data, ts[idx], scal)
            push!(klx, round(D_stsp, digits=5))
        end

        nlt_p = plot3d(og_data[:,1],og_data[:,2],og_data[:,3], lc=:blue, margin=0.01Plots.mm)
        plot3d!(nlt_p, ts_nlt[:,1], ts_nlt[:,2], ts_nlt[:,3], grid=true, 
                        xlabel="x", ylabel="y", zlabel="z", 
                        label=nothing, lc=:orange, legend=nothing, margin=0.01Plots.mm) 
        push!(plots, nlt_p)
        shallow_p = plot3d(og_data[:,1],og_data[:,2],og_data[:,3], lc=:blue, margin=0.01Plots.mm)
        plot3d!(shallow_p, ts_shallow[:,1], ts_shallow[:,2], ts_shallow[:,3], grid=true, 
                        xlabel="x", ylabel="y", zlabel="z", 
                        label=nothing, lc=:orange, legend=nothing, margin=0.01Plots.mm) 
        push!(plots, shallow_p)
    end
    comp_p = plot(plots[1], plots[2], plots[3], plots[4], margin=0.0Plots.mm,layout=grid(2,2),
     titlefont=font(8),title=["\n"*" "^70*"Inputs\nInside\nklx=$(klx[1])" "\n\nOutside\nklx=$(klx[2])" "\nklx=$(klx[3])" "\nklx=$(klx[4])"], plot_titlefont=font(14),plot_title="Comparison between Models")
    mkpath("Figures/evaluation/")
    savefig(comp_p, "Figures/evaluation/Compare_nlt_shallow_$run.png")
end

# evaluate_snapshots("Results/external_inputs/ExplodingLorenz_nlt/001/","last_model.bson", "nltPLRNN_exploding", "ExplodingLorenz", "data/snapshots/")
# evaluate_snapshots("Results/external_inputs/ShiftingLorenz_nlt/002/","last_model.bson", "nltPLRNN_Shifting", "ShiftingLorenz", "data/snapshots/")


compare_nlt_shallow("Results/external_inputs/Compare_nlt_shallow/", "ShiftingLorenz")
