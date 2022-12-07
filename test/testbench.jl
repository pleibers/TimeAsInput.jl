using NPZ
# using BPTT
using BSON: load
using TimeAsInput
using NNlib
using Flux

path="Results/external_inputs/Paper_base/model_4550.bson"
m = load(path, @__MODULE__)[:model]