using BPTT
using ArgParse


function args()
    settings = argtable()
    defaults = load_defaults()
    @add_arg_table! settings begin
        "--affine_transformation"
        help = "type of affine transformation"
        arg_type = String
        default = defaults["affine_transformation"]

        "--affine_transform_coeff"
        help = "coefficients for the transform function"
        arg_type = Vector{Any}
        default = defaults["affine_transform_coeff"]
    end
    return settings
end

commandline_parsing() = parse_args(args())
