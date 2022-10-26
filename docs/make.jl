using TimeAsInput
using Documenter

DocMeta.setdocmeta!(TimeAsInput, :DocTestSetup, :(using TimeAsInput); recursive=true)

makedocs(;
    modules=[TimeAsInput],
    authors="Patrick Leibersperger <pleibersperger@posteo.de> and contributors",
    repo="https://github.com/pleibers/TimeAsInput.jl/blob/{commit}{path}#{line}",
    sitename="TimeAsInput.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pleibers.github.io/TimeAsInput.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pleibers/TimeAsInput.jl",
    devbranch="main",
)
