#==
Grid-centric
==#

"""
    build_xygrid(x)

Return the 2 × n `Matrix` of `x.x` and `x.y` coordinates at which `x` is _not_ `NaN`.

In practice, this function can be used to return the grid on which the control points
are defined after localization has been applied to `x` by setting rejected entries to
`NaN`. `x` should only have named dimensions `x` and `y`, i.e. `x` passed to this function
is `x(:h)(t=0)(ens=1)`.

See also: [`densify`](@ref)
"""
function build_xygrid(x)
    gridshape = (length(x.y), length(x.x))
    xygrid = Matrix{Float64}(undef, 2, count(!isnan, x))
    CI = CartesianIndices(gridshape)
    idx = 1
    for i in eachindex(x)
        if !isnan(x[i])
            xygrid[:,idx] .= (x.x[CI[i][2]], x.y[CI[i][1]])
            idx += 1
        end
    end
    return xygrid
end

"""
    build_xygrid(mask, x_grid, y_grid)

Return the 2 × n `Matrix` of entries of `x_grid` and `y_grid` coordinates at which the mask
(usually from localization) is `true`.
"""
function build_xygrid(mask, x_grid, y_grid)
    gridshape = (length(y_grid), length(x_grid))
    xygrid = Matrix{Float64}(undef, 2, count(mask))
    CI = CartesianIndices(gridshape)
    idx = 1
    for i in eachindex(mask)
        if mask[i]
            xygrid[:,idx] .= (x_grid[CI[i][2]], y_grid[CI[i][1]])
            idx += 1
        end
    end
    return xygrid
end

"""
    build_xygrid(west, east, south, north, fromproj=wgs84(), toproj=esri_102010(); dr=300e3)

Return the `(x_grid, y_grid)` tuple of `StepRangeLen` in the `toproj` `Projection` from
`west` to `east` and `south` to `north` in the `fromproj` `Projection` using a step size
of `dr` in the model space.

`x_grid` begins at `west - 2dr` in model space and goes no further than `east + 2dr`. The
equivalent bounds are also applied to `y_grid`. In practice, localization is then applied
to this grid. We go out `2dr` instead of `1dr` to compensate for the fact that the true
distance may be larger than the projected `dr`.
"""
function build_xygrid(west, east, south, north, fromproj=wgs84(), toproj=esri_102010(); dr=300e3)
    trans = Proj.Transformation(fromproj, toproj)
    pts = [trans(west, north), trans(east, north), trans(west, south), trans(east, south)]
    xmin, xmax = extrema(getindex.(pts, 1))
    ymin, ymax = extrema(getindex.(pts, 2))

    # add `dr` because otherwise end will be previous value that is a multiple of dr
    x_grid = range(xmin-2dr, xmax+2dr; step=dr)
    y_grid = range(ymin-2dr, ymax+2dr; step=dr)

    return x_grid, y_grid
end

"""
    densify(x_grid, y_grid)

Return the ``2 × n`` matrix of all grid points over ranges `x_grid`, `y_grid` without filtering.

See also: [`build_xygrid`](@ref)
"""
function densify(x_grid, y_grid)
    return [(x, y) for x in x_grid for y in y_grid]
    # return reshape(reinterpret(Float64, [(x, y) for x in x_grid for y in y_grid]), 2, :)
end

"""
    dense_grid(itp, values, x_grid, y_grid)

Interpolate `values` over dense `x_grid`, `y_grid` in `itp.projection` and return a matrix
of size `(length(y_grid), length(x_grid))`.

!!! note

    For `itp.method` that specifies a field (common with `GeoStatsInterpolant`), the field
    should be `:v` when `itp` is passed to this function.
"""
function dense_grid(itp::ScatteredInterpolant, values, x_grid, y_grid)
    vitp = ScatteredInterpolation.interpolate(itp.method, itp.coords, filter(!isnan, values))

    xy_grid = densify(x_grid, y_grid)

    vgrid = Matrix{Float64}(undef, length(y_grid), length(x_grid))
    for i in axes(xy_grid,2)
        vgrid[i] = only(ScatteredInterpolation.evaluate(vitp, xy_grid[:,i]))
    end

    return vgrid
end

function dense_grid(itp::GeoStatsInterpolant, values, x_grid, y_grid)
    :v in itp.method.varnames ||
        throw(ArgumentError("`itp.method` should be defined for variable `:v` when passed to `dense_grid`."))

    geox = georef((v=filter(!isnan, values),), PointSet(itp.coords))

    xy_grid = densify(x_grid, y_grid)

    problem = EstimationProblem(geox, PointSet(xy_grid), :v)
    solution = solve(problem, itp.method)

    vgrid = Matrix{Float64}(undef, length(y_grid), length(x_grid))
    for i in axes(xy_grid,2)
        vgrid[i] = solution[:v][i]
    end

    return vgrid
end

#==
Localization-centric
==#

"""
    gaspari1999_410(z, c)

Compactly supported 5th-order piecewise rational function that resembles a Gaussian evaluated
over distances `z` with scale length `c`.

The length-scale ``L = 1 / (-f″(0))^{1/2}`` is ``c(0.3)^{1/2}``. The corresponding Gaussian
function is ``G(z, L) = exp(-z^2/(2L²))``.

See also: [`gaussianstddev`](@ref), [`compactlengthscale`](@ref)

# References

[^1]: Gaspari Cohn 1999, Construction of correlation functions in two and three dimensions.
    Eqn 4.10, Eqn 4.16
"""
function gaspari1999_410(z, c)
    C0 = zeros(size(z))

    for i in eachindex(C0)
        tz = z[i]
        if 0 <= abs(tz) <= c
            C0[i] = -(1/4)*(abs(tz)/c)^5 + (1/2)*(tz/c)^4 + (5/8)*(abs(tz)/c)^3 -
                (5/3)*(tz/c)^2 + 1
        elseif c <= abs(tz) <= 2c
            C0[i] = (1/12)*(abs(tz)/c)^5 - (1/2)*(tz/c)^4 + (5/8)*(abs(tz)/c)^3 +
                (5/3)*(tz/c)^2 - 5*(abs(tz)/c) + 4 - (2/3)*c/abs(tz)
        # elseif 2c <= abs(tz)
        # C0[i] = 0
        end
    end

    return C0
end

"""
    gaussianstddev(c)

Compute the standard deviation (length-scale) of a Gaussian function in terms of the
length-scale `c` of the compactly supported function [`gaspari1999_410`](@ref).

See also: [`compactlengthscale`](@ref), [`gaspari1999_410`](@ref)

# References

[^1]: Gaspari Cohn 1999, Construction of correlation functions in two and three dimensions.
    Eqn 4.12 and surrounding text.
"""
gaussianstddev(c) = c*sqrt(0.3)

"""
    compactlengthscale(σ)

Compute the compact length-scale `c` of the function [`gaspari1999_410`](@ref) in terms of
the standard deviation `σ` (also called ``L``) of a Gaussian function.

See also: [`gaussianstddev`](@ref), [`gaspari1999_410`](@ref)
"""
compactlengthscale(σ) = σ/sqrt(0.3)

"""
    lonlatgrid_dists(lonlats)

Compute distance in meters between every grid point in a matrix of `permutedims([lon lat])`
or vector of lon, lat tuples.
"""
function lonlatgrid_dists(lonlats)
    N = length(lonlats)
    distarr = Matrix{Float64}(undef, N, N)
    for j in eachindex(lonlats)
        for i in eachindex(lonlats)
            distarr[i,j] = inverse(lonlats[j]..., lonlats[i]...).dist
        end
    end
    return distarr
end

function lonlatgrid_dists(lonlats::Matrix)
    N = size(lonlats,2)
    distarr = Matrix{Float64}(undef, N, N)
    for j in axes(lonlats,2)
        for i in axes(lonlats,2)
            distarr[i,j] = inverse(lonlats[1,j], lonlats[2,j], lonlats[1,i], lonlats[2,i]).dist
        end
    end
    return distarr
end

"""
    obs2grid_diamondpill(lonlats, paths; overshoot=200e3, halfwidth=300e3) → (localization, diamonds)

Return a localization matrix of shape `(ngrid, npaths)` where `0.0` means the path does not
affect the grid cell or `1.0` meaning the path does.

`lonlats` is a vector of tuples of longitudes and latitude points.
`paths` is a vector of (transmitter, receiver) tuples representing each propagation path.

`diamonds` is a vector of vectors of points describing the localization pattern around each
path.

This function uses a localization shape that is shaped like a diamond that extends from the
transmitter to the receiver that widens to a width of `2halfwidth` meters in the middle.
The diamond actually overshoots the transmitter and receiver by `overshoot` meters and forms
a circle around the transmitter/receiver that joins with the diamond towards the
receiver/transmitter.

The localization is `1.0` if the grid point `intersects` the geometric Polygon of the diamond.

Compared to [`obs2grid_diamondcircle`](@ref), this function extends the diamond to the
minimum and maximum latitude extents of the receiver and transmitter circles and then takes
the convex hull of the points. Note that the convex hull is not necessarily the "correct"
solution because these points are on an ellipsoid and leads to an inappropriately large
region to the south in the northern hemisphere (it should really be a concave hull or other
"alpha" shape).

See also: [`obs2grid_diamondcircle`](@ref)
"""
function obs2grid_diamondpill(lonlats, paths; overshoot=200e3, halfwidth=300e3)
    ngrid = length(lonlats)
    npaths = length(paths)

    arc = 0:15:360-15

    localization = Matrix{Float64}(undef, ngrid, npaths)
    diamonds = Vector{Vector{Vector{Float64}}}()
    sizehint!(diamonds, npaths)
    # diamonds = []

    for p in eachindex(paths)
        tx = paths[p][1]
        rx = paths[p][2]

        # Get circle of points around transmitter
        tx_circ = [(pt = forward(tx.longitude, tx.latitude, az, overshoot); (pt.lon, pt.lat)) for az in arc]
        
        # Min and max latitude (only) of tx_circ
        tx_min = (Inf, Inf)
        tx_max = (-Inf, -Inf)
        for i in eachindex(tx_circ)
            if tx_circ[i][2] < tx_min[2]
                tx_min = tx_circ[i]
            elseif tx_circ[i][2] > tx_max[2]
                tx_max = tx_circ[i]
            end
        end

        # Get circle of points around receiver
        rx_circ = [(pt = forward(rx.longitude, rx.latitude, az, overshoot); (pt.lon, pt.lat)) for az in arc]
    
        # Min and max latitude (only) of rx_circ
        rx_min = (Inf, Inf)
        rx_max = (-Inf, -Inf)
        for i in eachindex(rx_circ)
            if rx_circ[i][2] < rx_min[2]
                rx_min = rx_circ[i]
            elseif rx_circ[i][2] > rx_max[2]
                rx_max = rx_circ[i]
            end
        end

        # Center point between rx and tx
        fwdaz, _, dist, _ = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude)
        center = forward(tx.longitude, tx.latitude, fwdaz, dist/2)

        midpt1 = forward(center.lon, center.lat, fwdaz+90, halfwidth)
        midpt2 = forward(center.lon, center.lat, fwdaz-90, halfwidth)
        if midpt1.lat > midpt2.lat
            uppermidpt = midpt1
            lowermidpt = midpt2
        else
            uppermidpt = midpt2
            lowermidpt = midpt1
        end

        # Great circle paths from rx_max to center to tx_max and rx_min to center to tx_min
        upper_gcp_rx = waypoints(GeodesicLine(rx_max...; lon2=uppermidpt.lon, lat2=uppermidpt.lat);
            n=100)
        upper_gcp_tx = waypoints(GeodesicLine(uppermidpt.lon, uppermidpt.lat; lon2=tx_max[1], lat2=tx_max[2]);
            n=100)
        lower_gcp_rx = waypoints(GeodesicLine(rx_min...; lon2=lowermidpt.lon, lat2=lowermidpt.lat);
            n=100)
        lower_gcp_tx = waypoints(GeodesicLine(lowermidpt.lon, lowermidpt.lat; lon2=tx_min[1], lat2=tx_min[2]);
            n=100)

        allpts = [
            [[pt[1], pt[2]] for pt in tx_circ];
            [[pt[1], pt[2]] for pt in rx_circ];
            [[pt.lon, pt.lat] for pt in upper_gcp_rx];
            [[pt.lon, pt.lat] for pt in upper_gcp_tx];
            [[pt.lon, pt.lat] for pt in lower_gcp_rx];
            [[pt.lon, pt.lat] for pt in lower_gcp_tx]
        ]
        diamond = LibGEOS.convexhull(LibGEOS.MultiPoint(allpts))

        # This sets to 1 if the gridcell is within the diamond at all and 0 otherwise
        for l in eachindex(lonlats)
            pt = LibGEOS.Point(lonlats[l]...)
            localization[l,p] = LibGEOS.intersects(pt, diamond) ? 1 : 0
        end

        # push!(diamonds, diamond)
        push!(diamonds, LibGEOS.coordinates(LibGEOS.boundary(diamond)))
    end

    return localization, diamonds
end


"""
    obs2grid_diamondcircle(lonlats, paths; overshoot=200e3, halfwidth=300e3) → (localization, diamonds)

Return a localization matrix of shape `(ngrid, npaths)` where `0.0` means the path does not
affect the grid cell or `1.0` meaning the path does.

`lonlats` is a vector of tuples of longitudes and latitude points.
`paths` is a vector of (transmitter, receiver) tuples representing each propagation path.

`diamonds` is a vector of vectors of points describing the localization pattern around each
path.

This function uses a localization shape that is shaped like a diamond that extends from the
transmitter to the receiver that widens to a width of `2halfwidth` meters in the middle.
The diamond actually overshoots the transmitter and receiver by `overshoot` meters and forms
a circle around the transmitter/receiver that joins with the diamond towards the
receiver/transmitter.

The localization is `1.0` if the grid point is contained within the geometric Polygon of
the diamond.

The difference between this function and [`obs2grid_diamondpill`](@ref) is that this
function strictly puts a circle around the transmitter and receiver that ends at +/- 90°
from the forward and back azimuths. This may lead to additional vertices where the circles
merge with the diamond. The diamond polygon is formed explicitly (without computing the
convex hull).

See also: [`obs2grid_diamondpill`](@ref)
"""
function obs2grid_diamondcircle(lonlats, paths; overshoot=200e3, halfwidth=300e3)
    ngrid = length(lonlats)
    npaths = length(paths)

    localization = Matrix{Float64}(undef, ngrid, npaths)
    diamonds = Vector{Vector{Vector{Float64}}}()
    sizehint!(diamonds, npaths)
    # diamonds = []

    for p in eachindex(paths)
        tx = paths[p][1]
        rx = paths[p][2]

        if tx.longitude > rx.longitude
            eastern = tx
            western = rx
        else
            eastern = rx
            western = tx
        end

        fwdaz, backaz, dist, _ = inverse(eastern.longitude, eastern.latitude, western.longitude, western.latitude)

        east_circ = Vector{Tuple{Float64,Float64}}(undef, 30)
        azrange = range(backaz-90, backaz+90; length=30)
        for i in eachindex(azrange)
            az = azrange[i]
            pt = forward(eastern.longitude, eastern.latitude, az, overshoot)
            east_circ[i] = (pt.lon, pt.lat)
        end

        west_circ = Vector{Tuple{Float64,Float64}}(undef, 30)
        azrange = range(fwdaz-90, fwdaz+90; length=30)
        for i in eachindex(azrange)
            az = azrange[i]
            pt = forward(western.longitude, western.latitude, az, overshoot)
            west_circ[i] = (pt.lon, pt.lat)
        end

        # Center point between tx and rx
        center = forward(eastern.longitude, eastern.latitude, fwdaz, dist/2)

        midpt1 = forward(center.lon, center.lat, fwdaz+90, halfwidth)
        midpt2 = forward(center.lon, center.lat, fwdaz-90, halfwidth)
        if midpt1.lat > midpt2.lat
            northmidpt = midpt1
            southmidpt = midpt2
        else
            northmidpt = midpt2
            southmidpt = midpt1
        end

        allpts = [[pt[1], pt[2]] for pt in east_circ]

        if east_circ[end][2] < east_circ[1][2]
            # If end of east_circ is the south side
            wpts = waypoints(GeodesicLine(east_circ[end]...; lon2=southmidpt.lon, lat2=southmidpt.lat); n=100)
            append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
            if west_circ[1][2] < west_circ[end][2]
                # If beginning of west_circ is also south
                wpts = waypoints(GeodesicLine(southmidpt.lon, southmidpt.lat; lon2=west_circ[1][1], lat2=west_circ[1][2]); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
                append!(allpts, [[pt[1], pt[2]] for pt in west_circ])
                wpts = waypoints(GeodesicLine(west_circ[end]...; lon2=northmidpt.lon, lat2=northmidpt.lat); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
            else
                wpts = waypoints(GeodesicLine(southmidpt.lon, southmidpt.lat; lon2=west_circ[end][1], lat2=west_circ[end][2]); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
                append!(allpts, [[pt[1], pt[2]] for pt in west_circ])
                wpts = waypoints(GeodesicLine(west_circ[1]...; lon2=northmidpt.lon, lat2=northmidpt.lat); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
            end
            append!(wpts, waypoints(GeodesicLine(northmidpt.lon, northmidpt.lat; lon2=east_circ[1][1], lat2=east_circ[1][2]); n=100))
        else
            # end of east_circ is north side
            wpts = waypoints(GeodesicLine(east_circ[end]...; lon2=northmidpt.lon, lat2=northmidpt.lat); n=100)
            append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
            if west_circ[1][2] > west_circ[end][2]
                # If beginning of west_circ is north side
                wpts = waypoints(GeodesicLine(northmidpt.lon, northmidpt.lat; lon2=west_circ[1][1], lat2=west_circ[1][2]); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
                append!(allpts, [[pt[1], pt[2]] for pt in west_circ])
                wpts = waypoints(GeodesicLine(west_circ[1]...; lon2=southmidpt.lon, southmidpt.lat); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
            else
                wpts = waypoints(GeodesicLine(northmidpt.lon, northmidpt.lat; lon2=west_circ[end][1], lat2=west_circ[end][2]); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
                append!(allpts, [[pt[1], pt[2]] for pt in west_circ])
                wpts = waypoints(GeodesicLine(west_circ[end]...; lon2=southmidpt.lon, southmidpt.lat); n=100)
                append!(allpts, [[pt.lon, pt.lat] for pt in wpts])
            end
        end

        unique!(allpts)
        push!(allpts, allpts[1])  # first = end to close LineString
        
        diamond = LibGEOS.Polygon([allpts])

        # This sets to 1 if the gridcell is within the diamond at all and 0 otherwise
        for l in eachindex(lonlats)
            pt = LibGEOS.Point(lonlats[l]...)
            localization[l,p] = LibGEOS.contains(diamond, pt) ? 1 : 0
        end

        # push!(diamonds, diamond)
        push!(diamonds, LibGEOS.coordinates(LibGEOS.boundary(diamond)))
    end

    return localization, diamonds
end

"""
    boundary_coords(paths)

Return great circle paths along the points of the convex hull over the propagation paths as
well as points along every path `(gcp_boundary, wpts)`.
"""
function boundary_coords(paths)
    wpts = Vector{Vector{Float64}}()
    for p in paths
        tx, rx = p[1], p[2]
        _, wp = pathpts(tx, rx; dist=10e3)
        for i in eachindex(wp)
            push!(wpts, [wp[i].lon, wp[i].lat])
        end
    end
    mpts = LibGEOS.MultiPoint(wpts)
    hull = LibGEOS.convexhull(mpts)

    # for plotting...
    # trans = Proj.Transformation(wgs84(), model_projection)
    # pwpts = trans.(getindex.(wpts, 1), getindex.(wpts, 2))

    # Remove very close points
    hull = LibGEOS.simplify(hull, 0.1)

    # Get GCP pts along convex hull (boundary)
    hull_coords = only(LibGEOS.coordinates(hull))::Vector{Vector{Float64}}
    gcp_boundary = Vector{Tuple{Float64,Float64}}()
    for i in 1:length(hull_coords)-1
        h = hull_coords[i]
        line = GeographicLib.GeodesicLine(h[1], h[2];
            lon2=hull_coords[i+1][1], lat2=hull_coords[i+1][2])
        wp = waypoints(line; dist=10e3)
        for w in wp
            push!(gcp_boundary, (w.lon, w.lat))
        end
    end

    return gcp_boundary, wpts
end

"""
    obs2grid_distance(lonlats, paths, r=200e3, pathstep=100e3)

Return `localization` matrix that identifies whether or not each element of `lonlats` is
within `r` meters of each path.

See also: [`localize_distance`](@ref)
"""
function obs2grid_distance(lonlats, paths; r=200e3, pathstep=100e3)
    ngrid = length(lonlats)
    npaths = length(paths)

    localization = trues(ngrid, npaths)
    for p in eachindex(paths)
        tx, rx = paths[p][1], paths[p][2]
        _, wpts = pathpts(tx, rx; dist=pathstep)

        # `wpts[1]` is at the transmitter. For the purposes of localization, begin with the
        # next waypoint after the transmitter. (`wpts[end]` is before the receiver)
        wpts = wpts[2:end]

        for j in eachindex(lonlats)
            lo, la = lonlats[j]
            dmin = Inf
            for i in eachindex(wpts)
                d = inverse(lo, la, wpts[i].lon, wpts[i].lat).dist
                if d < dmin
                    dmin = d
                end
            end
            
            if dmin > r
                localization[j,p] = false
            end
        end
    end

    return localization
end

"""
    obs2grid_distances(lonlats, paths; pathstep=100e3)

Return matrix of actual distances from each path to each grid point.

Like `obs2grid_distance` except it  returns actual distances instead of the result.
TODO: combine these two functions.
"""
function obs2grid_distances(lonlats, paths; pathstep=100e3)
    ngrid = length(lonlats)
    npaths = length(paths)

    distances = Matrix{Float64}(undef, ngrid, npaths)
    for p in eachindex(paths)
        tx, rx = paths[p][1], paths[p][2]
        _, wpts = pathpts(tx, rx; dist=pathstep)

        # `wpts[1]` is at the transmitter. For the purposes of localization, begin with the
        # next waypoint after the transmitter. (`wpts[end]` is before the receiver)
        wpts = wpts[2:end]

        for j in eachindex(lonlats)
            lo, la = lonlats[j]
            dmin = Inf
            for i in eachindex(wpts)
                d = inverse(lo, la, wpts[i].lon, wpts[i].lat).dist
                if d < dmin
                    dmin = d
                end
            end
            
            distances[j,p] = dmin
        end
    end

    return distances
end

"""
    filterbounds!(localization, lonlat, west, east, south, north)

Set `localization` entries to `0` if the corresponding `lonlat` entry is outside of the
rectangular region bounded by `west`, `east`, `south`, `north`.
"""
function filterbounds!(localization, lonlat, west, east, south, north)
    for i in eachindex(lonlat)
        if lonlat[i][1] < west || lonlat[i][1] > east || lonlat[i][2] < south || lonlat[i][2] > north
            localization[i,:] .= 0
        end
    end
    return localization
end


"""
    anylocal(localization)

Convenience function that returns a `Vector{Bool}` of whether or not there is any
localization in any path for a matrix `localization` of size `(ngrid, npaths)`.

See also: [`obs2grid_distance`](@ref), [`obs2grid_diamondpill`](@ref)
"""
function anylocal(localization)
    localize = trues(size(localization,1))
    for i in axes(localization,1)
        # Check if not a single path affects gridcell i
        if all(x->x==0, localization[i,:])
            localize[i] = false
        end
    end
    return localize
end

"""
    mediandr(lola)

Return the median WGS84 distance in meters between dense matrix of longitude, latitude points
in ``2 × n`` `lola`.
"""
function mediandr(lola)
    dists = Vector{Float64}(undef, length(lola)÷2)
    idx = 1
    for i = 1:2:length(lola)-1
        dists[idx] = inverse(lola[i]..., lola[i+1]...).dist
        idx += 1
    end
    return median(dists)
end

function mediandr(lola::Matrix)
    dists = Vector{Float64}(undef, size(lola,2)÷2)
    idx = 1
    for i = 1:2:size(lola,2)-1
        dists[idx] = inverse(lola[1,i],lola[2,i],lola[1,i+1],lola[2,i+1]).dist
        idx += 1
    end
    return median(dists)
end
