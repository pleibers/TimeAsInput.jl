using Benchmarks: TP_loc

Base.Float32(V::AbstractVector) = Base.Float32.(V)
"""
    main_routine(args)

Function executed by every worker process.

main routine for training a plrnn, here with external inputs
"""
function main_training_routine(args::AbstractDict)
    # num threads
    n_threads = Threads.nthreads()
    println("Running on $n_threads Thread(s)")

    # get computing device
    device = get_device(args)
    if contains(args["path_to_data"], "Paper")
        args["path_to_inputs"] = "data/time_data/time_PaperLorenzBigChange.npy"
    elseif contains(args["path_to_data"], "Stop")
        args["path_to_inputs"] = "data/time_data/time_StopBurstBN.npy"
    end
    println(args["path_to_inputs"])
    # check if external inputs are provided
    if !isempty(args["path_to_inputs"])
        println("Path to external inputs provided, initializing ExternalInputsDataset.")
        D = ExternalInputsDataset(
            args["path_to_data"],
            args["path_to_inputs"];
            device = device,
        )
    else
        println("No path to external inputs provided, initializing vanilla Dataset.")
        D = Dataset(args["path_to_data"]; device = device)
    end

    # do the affine transformation of time, as only time itself is loaded
    if typeof(D) <: ExternalInputsDataset
        if args["prediction"]
            D, D_test = train_test_split(D, TP_loc[get_model_from_path(args["path_to_data"])])
        end
    end
    
    # model
    plrnn = initialize_model(args, D;mod=@__MODULE__) |> device

    if typeof(plrnn) <: ptPLRNN
        plrnn.t = D.S
    end

    # observation_model
    O = initialize_observation_model(args, D) |> device
    
    # optimizer
    opt = initialize_optimizer(args)

    # create directories
    save_path = create_folder_structure(args["experiment"], args["name"], args["run"])

    # store hypers
    store_hypers(args, save_path)

    train_!(plrnn, O, D, opt, args, save_path)
end
