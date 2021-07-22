using Dates
using Plots
using AxisKeys, ScatteredInterpolation, GeoStats
using Proj4
using LongwaveModePropagator
using LMPTools

using SubionosphericVLFInversionAlgorithms
const SIA = SubionosphericVLFInversionAlgorithms

function gasparicohn()
    z = 0:2000e3
    c = 1000e3
    gc = gaspari_cohn99_410(z, c)

    plot(z, gc)
end

function buildpaths()
    transmitters = [TRANSMITTER[:NLK], TRANSMITTER[:NML]]
    receivers = [
        Receiver("Whitehorse", 60.724, -135.043, 0.0, VerticalDipole()),
        Receiver("Churchill", 58.74, -94.085, 0.0, VerticalDipole()),
        Receiver("Stony Rapids", 59.253, -105.834, 0.0, VerticalDipole()),
        Receiver("Fort Smith", 60.006, -111.92, 0.0, VerticalDipole()),
        Receiver("Bella Bella", 52.1675508, -128.1545219, 0.0, VerticalDipole()),
        Receiver("Nahanni Butte", 61.0304412, -123.3926734, 0.0, VerticalDipole()),
        Receiver("Juneau", 58.32, -134.41, 0.0, VerticalDipole()),
        Receiver("Ketchikan", 55.35, -131.673, 0.0, VerticalDipole()),
        Receiver("Winnipeg", 49.8822, -97.1308, 0.0, VerticalDipole()),
        Receiver("IslandLake", 53.8626, -94.6658, 0.0, VerticalDipole()),
        Receiver("Gillam", 56.3477, -94.7093, 0.0, VerticalDipole())
    ]
    paths = [(tx, rx) for tx in transmitters for rx in receivers]

    return paths
end

function day(lo, la)
    dt = DateTime(2020, 3, 1, 20, 00)
    ferguson(la, zenithangle(la, lo, dt), dt)
end

function pert(lo, la)
    dt = DateTime(2020, 3, 1, 2, 0)
    coeffh = (0.0, -0.1, 0.0, 0.1, 0.0)
    coeffb = (0.0, -0.01, 0.0, 0.01, 0.0)
    sza = zenithangle(la, lo, dt)
    hpert = fourierperturbation(sza, coeffh)
    bpert = fourierperturbation(sza, coeffb)

    return hpert, bpert
end
# hmap, bmap, x_grid, y_grid = truth(pert)
# heatmap(x_grid, y_grid, bmap;
#         color=:amp, xlims=extrema(x_grid), ylims=extrema(y_grid))

function terminator(lo, la)
    dt = DateTime(2020, 3, 1, 2, 0)
    flatlinearterminator(zenithangle(la, lo, dt))
end

function sterminator(lo, la)
    dt = DateTime(2020, 3, 1, 2, 0)
    smoothterminator(zenithangle(la, lo, dt); steepness=0.8)
end

function bumpyday(lo, la)
    dt = DateTime(2020, 3, 1, 20, 00)
    coeffh = (1.3, 1.6, -0.1, 2.8, -3.3)
    coeffb = (0.075, 0.098, 0.031, -0.049, 0.098)
    sza = zenithangle(la, lo, dt)
    h, b = ferguson(la, sza, dt)
    hpert = fourierperturbation(sza, coeffh)
    bpert = fourierperturbation(sza, coeffb)
    return h+hpert, b+bpert
end

function bounds()
    west, east = -135.5, -93
    south, north = 46, 63

    return west, east, south, north
end

function truth(f)
    w, e, s, n = bounds()
    modelproj = esri_102010()
    
    x_grid, y_grid = build_xygrid(w, e, s, n, wgs84(), modelproj; dr=20e3)
    xy_grid = collect(densify(x_grid, y_grid))
    lola = permutedims(transform(modelproj, wgs84(), permutedims(xy_grid)))

    hmap = Matrix{Float64}(undef, length(y_grid), length(x_grid))
    bmap = similar(hmap)
    for i in axes(lola, 2)
        h, b = f(lola[:,i]...)
        hmap[i] = h
        bmap[i] = b
    end
    return hmap, bmap, x_grid, y_grid
end

function plottruth(f)
    hmap, bmap, x_grid, y_grid = truth(f)

    heatmap(x_grid, y_grid, hmap; clims=clims(f),
        color=:amp, xlims=extrema(x_grid), ylims=extrema(y_grid))
end

function clims(f)
    if f == day
        clims = (71, 75)
    elseif f == terminator || f == sterminator
        clims = (71, 87)
    elseif f == bumpyday
        clims = (69, 80)
    end
    return clims
end

function scattereditp(f)
    w, e, s, n = bounds()
    modelproj = esri_102010()
    paths = buildpaths()

    dr = 200e3
    lengthscale = 1200e3
    x_grid, y_grid = build_xygrid(w, e, s, n, wgs84(), modelproj; dr)

    gridshape = (length(y_grid), length(x_grid))

    xy_grid = collect(densify(x_grid, y_grid))
    lola = permutedims(transform(modelproj, wgs84(), permutedims(xy_grid)))

    truedr = SIA.mediandr(lola)
    modelscale = truedr/dr
    
    h = Matrix{Float64}(undef, gridshape...)
    for i in axes(lola,2)
        h[i] = f(lola[:,i]...)[1]
    end

    localization, _ = obs2grid_diamondpill(lola, paths;
        overshoot=sqrt(2)*dr*modelscale, halfwidth=lengthscale/2)
    locmask = anylocal(localization)

    h[.!locmask] .= NaN

    itppts = build_xygrid(KeyedArray(h; y=y_grid, x=x_grid))
    itp = ScatteredInterpolant(ThinPlate(), modelproj, itppts)

    trueh, trueb, fine_xgrid, fine_ygrid = truth(f)
    hgrid = dense_grid(itp, h, fine_xgrid, fine_ygrid)
    errgrid = hgrid .- trueh

    h1 = heatmap(fine_xgrid, fine_ygrid, hgrid; clims=clims(f), cbar=false,
        color=:amp, xlims=extrema(fine_xgrid), ylims=extrema(fine_ygrid))
    scatter!(h1, itppts[1,:], itppts[2,:];
        zcolor=filter(!isnan, h), color=:amp, markerstrokecolor="black", legend=false)
    h2 = heatmap(fine_xgrid, fine_ygrid, errgrid; clims=(-0.2, 0.2),
        color=:coolwarm, xlims=extrema(fine_xgrid), ylims=extrema(fine_ygrid), yticks=nothing)
    scatter!(h2, itppts[1,:], itppts[2,:]; color=:black, legend=false)
    plot(h1, h2; size=(1100,600), layout=grid(1,2,widths=[0.45,0.55]))
end

function geostatsitp(f, dr=300e3, lengthscale=1200e3, τ=9e-8*lengthscale)
    w, e, s, n = bounds()
    modelproj = esri_102010()
    paths = buildpaths()

    x_grid, y_grid = build_xygrid(w, e, s, n, wgs84(), modelproj; dr)

    gridshape = (length(y_grid), length(x_grid))

    xy_grid = collect(densify(x_grid, y_grid))
    lola = permutedims(transform(modelproj, wgs84(), permutedims(xy_grid)))
    
    h = Matrix{Float64}(undef, gridshape...)
    b = similar(h)
    for i in axes(lola,2)
        hp, be = f(lola[:,i]...)
        h[i] = hp
        b[i] = be
    end

    localization = obs2grid_distance(lola, paths; r=lengthscale/2)
    locmask = anylocal(localization)

    h[.!locmask] .= NaN
    b[.!locmask] .= NaN

    itppts = build_xygrid(KeyedArray(h; y=y_grid, x=x_grid))

    # τ = 1e-7*lengthscale
    solver = LWR(
        :h′ => (weightfun=r->exp(-r^2/(2*τ^2)),),
        :β => (weightfun=r->exp(-r^2/(2*τ^2)),),
        :v => (weightfun=r->exp(-r^2/(2*τ^2)),)
    )

    itp = GeoStatsInterpolant(solver, modelproj, itppts)

    trueh, trueb, fine_xgrid, fine_ygrid = truth(f)
    hgrid = dense_grid(itp, h, fine_xgrid, fine_ygrid)
    bgrid = dense_grid(itp, b, fine_xgrid, fine_ygrid)
    herrgrid = hgrid .- trueh
    berrgrid = bgrid .- trueb

    # plotgrid = hgrid
    # ploterrgrid = herrgrid
    # pclims = (71, 87)
    # pcol = :amp
    # eclims = (-0.2, 0.2)

    plotgrid = bgrid
    ploterrgrid = berrgrid
    pclims = (0.2, 0.6)
    pcol = :tempo
    eclims = (-0.01, 0.01)

    h1 = heatmap(fine_xgrid, fine_ygrid, plotgrid; clims=pclims, cbar=false,
        color=pcol, xlims=extrema(fine_xgrid), ylims=extrema(fine_ygrid))
    scatter!(h1, itppts[1,:], itppts[2,:];
        zcolor=filter(!isnan, h), color=pcol, markerstrokecolor="black", legend=false)
    h2 = heatmap(fine_xgrid, fine_ygrid, ploterrgrid; clims=eclims,
        color=:coolwarm, xlims=extrema(fine_xgrid), ylims=extrema(fine_ygrid), yticks=nothing)
    scatter!(h2, itppts[1,:], itppts[2,:]; color=:black, legend=false)
    plot(h1, h2; size=(1100,600), layout=grid(1,2,widths=[0.45,0.55]))
end
