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
    var = false
    try 
        args["optional_model_args"][1]
        var = true
    catch
    end
    if contains(args["path_to_data"], "Paper") && !var && args["model"] != "nltPLRNN"
        args["path_to_inputs"] = "data/time_data/time_PaperLorenzBigChange.npy"
        args["weak_tf_alpha"] = 0.1f0
        println("Attention: defaults were changed")
    elseif contains(args["path_to_data"], "Stop") && !var && args["model"] != "nltPLRNN"
        args["path_to_inputs"] = "data/time_data/time_StopBurstBN.npy"
        args["weak_tf_alpha"] = 0.5f0
        println("Attention: defaults were changed")
    end
    println(args["path_to_inputs"])

    run = true
    if args["lat_model_regularization"] != 0.0f0
        if "ar" in args["optional_model_args"] || "mlp" in args["optional_model_args"]
            run = false
        end
    end
    # check if external inputs are provided
    if !isempty(args["path_to_inputs"])
        println("Path to external inputs provided, initializing ExternalInputsDataset.")
        D = ExternalInputsDataset(
            args["path_to_data"],
            args["path_to_inputs"];
            device=device
        )
    else
        println("No path to external inputs provided, initializing vanilla Dataset.")
        D = Dataset(args["path_to_data"]; device=device)
    end

    # do the affine transformation of time, as only time itself is loaded
    if typeof(D) <: ExternalInputsDataset
        if args["prediction"]
            D, D_test = train_test_split(D, TP_loc[get_model_from_path(args["path_to_data"])])
        end
    end


    run = args["run_anyway"] ? true : run
    if run
        # model
        plrnn = initialize_model(args, D; mod=@__MODULE__) |> device

        if typeof(plrnn) <: AbstractNSPLRNN
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
    else
        println("-"^50)
        println("not running because of memory")
        println("-"^50)
    end
end
