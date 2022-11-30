using NPZ
using BPTT
using BSON

path="Results/external_inputs/StopBurstBN_mlp/last_model.bson"
m = load(path, @__MODULE__)