using Documenter
using DensityFlows

DocMeta.setdocmeta!(
    DensityFlows,
    :DocTestSetup,
    :(using DensityFlows, Flux);
    recursive=true
)

pages = [
    "Overview" => "index.md",
    "Manual" => ["manual.md", "example.md"],
    "Public API" => ["api_overview.md", "api_coupling.md", "api_data.md", "api_flow.md"]
]

makedocs(
    ;modules=[DensityFlows], 
    sitename="DensityFlows.jl", 
    pages=pages,
    meta=Dict(:DocTestSetup => :(using DensityFlows, Flux)),
    format = Documenter.HTML(collapselevel = 1)
    )