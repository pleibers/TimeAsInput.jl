#!/bin/bash

julia -t 2 --project uber_training.jl -p 15
wait $!
julia -t 5 --project main_training.jl --run_anyway true --lat_model_regularization 0.1 --name lat_nsw_0.1 --model nswPLRNN
wait $!
julia -t 5 --project main_training.jl --run_anyway true --lat_model_regularization 0.3 --name lat_nsw_0.3 --model nswPLRNN
