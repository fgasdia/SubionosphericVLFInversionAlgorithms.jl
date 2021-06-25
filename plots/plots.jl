using Plots

using LMPTools
using SubionosphericVLFInversionAlgorithms
const SIA = SubionosphericVLFInversionAlgorithms

function gasparicohn()
    z = 0:2000e3
    c = 1000e3
    gc = gaspari_cohn99_410(z, c)

    plot(z, gc)
end

function thinplatespline()
    include(SIA.project_path(joinpath("test", "utils.jl")))
    x, _, _ = testscenario()
    gridshape = (length(x.y), length(x.x))

    xygrid = SIA.build_xygrid(x)

    lola = permutedims(transform(esri_102010(), wgs84(), permutedims(xygrid)))

    dt = DateTime(2020, 3, 2, 0, 30)  # terminator
    iono = [flatlinearterminator(zenithangle(lola[2,i], lola[1,i], dt)) for i in axes(lola,2)]
    x(:h) .= reshape(getindex.(iono,1), gridshape)
    x(:b) .= reshape(getindex.(iono,2), gridshape)

    itp = ScatteredInterpolation.interpolate(ThinPlate(), xygrid, filter(!isnan, x(:h)))

    ygrid = minimum(x.y):step(x.y)/3:maximum(x.y)
    xgrid = minimum(x.x):step(x.x)/3:maximum(x.x)
    densegrid = Matrix{Float64}(undef, 2, length(ygrid)*length(xgrid))
    idx = 1
    for j in eachindex(xgrid)
        for i in eachindex(ygrid)
            densegrid[:,idx] .= (xgrid[j], ygrid[i])
            idx += 1
        end
    end
    v = ScatteredInterpolation.evaluate(itp, densegrid)
    h1 = heatmap(xgrid, ygrid, reshape(v, length(ygrid), length(xgrid)), clims=(73, 88), color=:amp,
        xlims=(minimum(xgrid), maximum(xgrid)), ylims=(minimum(ygrid), maximum(ygrid)))
    h2 = heatmap(x.x, x.y, x(:h), clims=(73, 88), color=:amp,
        xlims=(minimum(xgrid), maximum(xgrid)), ylims=(minimum(ygrid), maximum(ygrid)))
    plot(h1, h2; size=(1200,600))
end
