function get_name(run::String)
    pm = "affine"
    model = "pt"
    hidden_dim = "200"
    lat_reg = "0.0"
    act_fun = "relu"
    data = "paper"
    if contains(run, "mlp")
        pm = "mlp"
    elseif contains(run, "affine")
        pm = "affine"
    elseif contains(run, "ar")
        pm = "ar"
    end

    if contains(run, "pt")
        model = "pt"
    elseif contains(run, "nsw")
        model = "nsw"
    end

    if contains(run, "200")
        hidden_dim = "200"
    elseif contains(run, "600")
        hidden_dim = "600"
    end

    if contains(run, "0.0")
        lat_reg = "0.0"
    elseif contains(run, "0.1")
        lat_reg = "0.1"
    elseif contains(run, "0.3")
        lat_reg = "0.3"
    end

    if contains(run, "relu")
        act_fun = "relu"
    elseif contains(run, "tanh")
        act_fun = "tanh"
    end

    if contains(run, "Paper")
        data = "PaperLorenzBigChange"
    elseif contains(run, "StopBurstBN")
        data = "StopBurstBN"
    end

    name = "$(data)_$(model)_$(hidden_dim)_$(pm)_$(act_fun)_$(lat_reg)"
    return name, data
end


function plot3dseries(tseries, title, t; legend=false)
    l_b = legend ? true : nothing
    plot3d(tseries[:, 1], tseries[:, 2], tseries[:, 3], title=title, lc=(:viridis), linez=t, xlabel="X", ylabel="Y", zlabel="Z", label=nothing, legend=l_b)
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

function gen_at_t(plrnn::AbstractShallowPLRNN, t::Int, time::AbstractMatrix, start)
    time_input = similar(time)
    time_input .= time[t, 1]
    # maybe need to adjust load model
    tseries = generate(plrnn, start, time_input, size(time_input, 1))
    return tseries
end