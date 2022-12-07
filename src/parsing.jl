using ArgParse

function args_table()
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

        "--prediction"
        help = "train on predition task"
        arg_type = Bool
        default = defaults["prediction"]
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

commandline_parsing() = parse_args(args_table())
