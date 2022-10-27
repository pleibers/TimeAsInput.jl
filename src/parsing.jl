using BPTT
using ArgParse

function initialize_model(args::AbstractDict, D::Dataset)
    # gather args
    N = size(D.X, 2)
    M = args["latent_dim"]

    @assert N <= M "Model latent dimension M=$M has to be at least
        as large as the dimension of the data N=$(N)!"

    B = args["num_bases"]
    Layers = args["hidden_layers"]
    model_name = args["model"]
    hidden_dim = args["hidden_dim"]

    if model_name == "PLRNN"
        model = PLRNN(M, N)
    elseif model_name == "mcPLRNN"
        model = mcPLRNN(M, N)
    elseif model_name == "clippedPLRNN"
        model = clippedPLRNN(M, N, B, D.X)
    elseif model_name == "FCDendPLRNN"
        model = FCDendPLRNN(M, N, B, D.X)
    elseif model_name == "deepPLRNN"
        model = deepPLRNN(M, Layers, N)
    elseif model_name == "shallowPLRNN"
        model = shallowPLRNN(M, hidden_dim, N)
    elseif model_name == "nltPLRNN"
        model = nltPLRNN(M,hidden_dim,N)
    end

    println("Model / # Parameters: $(typeof(model)) / $(num_params(model))")
    return model
end

function initialize_model(args::AbstractDict, D::ExternalInputsDataset)
    # gather args
    N = size(D.X, 2)
    K = size(D.S, 2)
    M = args["latent_dim"]

    @assert N <= M "Model latent dimension M=$M has to be at least
        as large as the dimension of the data N=$(N)!"

    B = args["num_bases"]
    Layers = args["hidden_layers"]
    model_name = args["model"]
    hidden_dim = args["hidden_dim"]

    if model_name == "PLRNN"
        model = PLRNN(M, N, K)
    elseif model_name == "mcPLRNN"
        model = mcPLRNN(M, N, K)
    elseif model_name == "clippedPLRNN"
        model = clippedPLRNN(M, N, B, D.X, K)
    elseif model_name == "FCDendPLRNN"
        model = FCDendPLRNN(M, N, B, D.X, K)
    elseif model_name == "deepPLRNN"
        model = deepPLRNN(M, Layers, N, K)
    elseif model_name == "shallowPLRNN"
        model = shallowPLRNN(M, hidden_dim, N, K)
    elseif model ="nltPLRNN"
        model = nltPLRNN(M,hidden_dim,N,K)
    end

    println("Model / # Parameters: $(typeof(model)) / $(num_params(model))")
    return model
end


function args()
    settings = argtable()
    defaults = load_defaults()
    @add_arg_table! settings begin
        "--affine_transformation"
        help = "type of affine transformation"
        arg_type = String
        default = defaults["affine_transformation"] |> String

        "--affine_transform_coeff"
        help = "coefficients for the transform function"
        arg_type = Vector
        default = defaults["affine_transform_coeff"]
    end
    return settings
end

function parse_transform(a_t::String, coefficients::AbstractVector)
    if a_t == "linear"
        return t -> linear(coefficients..., t)
    else
        throw("Unsupported transformation $a_t")
    end
end

commandline_parsing() = parse_args(args())
