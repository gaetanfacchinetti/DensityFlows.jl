using Documenter
using DensityFlows

DocMeta.setdocmeta!(
    DensityFlows,
    :DocTestSetup,
    :(using DensityFlows);
    recursive=true
)

pages = [
    "Overview" => "index.md",
    "Manual" => "manual.md",
    "Public API" => "api.md"
]

makedocs(
    ;modules=[DensityFlows], 
    sitename="DensityFlows.jl", 
    pages=pages,
    meta=Dict(:DocTestSetup => :(using DensityFlows)),
    format = Documenter.HTML(collapselevel = 1)
    )