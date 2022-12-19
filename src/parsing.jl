using ArgParse

function args_table()
    settings = argtable()
    defaults = load_defaults()
    @add_arg_table! settings begin
        "--prediction"
        help = "train on predition task"
        arg_type = Bool
        default = defaults["prediction"]
    end
    return settings
end

commandline_parsing() = parse_args(args_table())
