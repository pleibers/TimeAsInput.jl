using NPZ
# normalize(v::AbstractVector) = 2 .* (v .- minimum(v)) ./ (maximum(v) - minimum(v)) .- 1

# T = 800
# ΔT =0.01
# name = "PaperLorenzBigChange"

# t = 0.0:ΔT:T
# time = normalize(collect(t))
# time = reshape(time,length(time),1)

# mkpath("data/time_data")
# npzwrite("data/time_data/time_$name.npy", time)
# npzwrite("data/time_data/realtime_$name.npy", t)
# t = npzread("data/time_data/time_PaperLorenzBigChange.npy")
# size(t)