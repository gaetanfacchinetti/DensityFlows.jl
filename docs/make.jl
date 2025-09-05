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
    "Manual" => "index.md",
    "Public API" => ["AffineCoupling" => "api.md"]
]

makedocs(;modules=[DensityFlows], sitename="DensityFlows", pages=pages, meta=Dict(:DocTestSetup => :(using DensityFlows)))