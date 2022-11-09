using BPTT

Base.Float32(V::Vector{Float64}) = Base.Float32.(V)
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
        trans_coeff = args["affine_transform_coeff"]
        affine_transformation = parse_transform(args["affine_transformation"], trans_coeff)

        external_inputs = Float32.(affine_transformation.(D.S[:,1])) # to have type consistency
        ext_in = permutedims(reduce(hcat, external_inputs), (2,1))
        D = ExternalInputsDataset(D.X, ext_in,"time_in")
    end

    # model
    plrnn = initialize_model(args, D) |> device

    # optimizer
    opt = initialize_optimizer(args)

    # create directories
    save_path = create_folder_structure(args["experiment"], args["name"], args["run"])

    # store hypers
    store_hypers(args, save_path)

    train_!(plrnn, D, opt, args, save_path)
end
