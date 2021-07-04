using Plots
using AxisKeys
using LMPTools

using SubionosphericVLFInversionAlgorithms
const SIA = SubionosphericVLFInversionAlgorithms

function gasparicohn()
    z = 0:2000e3
    c = 1000e3
    gc = gaspari_cohn99_410(z, c)

    plot(z, gc)
end

function day(lo, la)
    dt = DateTime(2020, 3, 1, 20, 00)
    ferguson(la, zenithangle(la, lo, dt), dt)
end

function terminator(lo, la)
    dt = DateTime(2020, 3, 1, 2, 0)
    flatlinearterminator(zenithangle(la, lo, dt))
end

function bumpy(lo, la)
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
    elseif f == terminator
        clims = (71, 87)
    elseif f == bumpy
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
    h2 = heatmap(fine_xgrid, fine_ygrid, errgrid; clims=(-0.05, 0.05),
        color=:coolwarm, xlims=extrema(fine_xgrid), ylims=extrema(fine_ygrid), yticks=nothing)
    scatter!(h2, itppts[1,:], itppts[2,:]; color=:black, legend=false)
    plot(h1, h2; size=(1100,600), layout=grid(1,2,widths=[0.45,0.55]))
end
