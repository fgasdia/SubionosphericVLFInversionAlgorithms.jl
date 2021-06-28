"""
    gaspari1999_410(z, c)

Compactly supported 5th-order piecewise rational function that resembles a Gaussian evaluated
over distances `z` with scale length `c`.

# References

[^1]: Gaspari Cohn 1999, Construction of correlation functions in two and three dimensions.
    Eqn 4.10
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
    lonlatgrid_dists(lonlats)

Compute distance in meters between every grid point in a matrix of `permutedims([lon lat])`.
"""
function lonlatgrid_dists(lonlats)
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

`lonlats` is a dense matrix of longitudes and latitude points.
`paths` is a vector of (transmitter, receiver) tuples representing each propagation path.

`diamonds` is a vector of vectors of points describing the localization pattern around each
path.

This function uses a localization shape that is shaped like a diamond that extends from the
transmitter to the receiver that widens to a width of `2halfwidth` meters in the middle.
The diamond actually overshoots the transmitter and receiver by `overshoot` meters and forms
a circle around the transmitter/receiver that joins with the diamond towards the
receiver/transmitter.

The localization is `1.0` if the grid point `intersects` the geometric Polygon of the diamond. 
"""
function obs2grid_diamondpill(lonlats, paths; overshoot=200e3, halfwidth=300e3)
    # transmitters, receivers, _ = paths()
    ngrid = size(lonlats, 2)
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
        for l in axes(lonlats,2)
            pt = LibGEOS.Point(lonlats[1,l], lonlats[2,l])
            localization[l,p] = LibGEOS.intersects(pt, diamond) ? 1 : 0
        end

        # push!(diamonds, diamond)
        push!(diamonds, LibGEOS.coordinates(LibGEOS.boundary(diamond)))
    end

    return localization, diamonds
end

"""
    pathname(p)

Return path name string for (transmitter, receiver) path tuple `p`.
"""
pathname(p) = p[1].name*"-"*p[2].name

"""
    phasediff(a, b; deg=false)

Compute the smallest angle `a - b` in radians if `deg=false`, otherwise degrees.
"""
function phasediff(a, b; deg=false)
    if deg
        a, b = deg2rad(a), deg2rad(b)
    end

    d = mod2pi(a) - mod2pi(b)
    d = mod2pi(d + π) - π

    if deg
        d = rad2deg(d)
    end

    return d
end

"""
    strip(m::KeyedArray)
    strip(m::NamedDimsArray)

Remove named dims and axis keys from `m`, returning a view of the underlying array.
"""
Base.strip(m::KeyedArray) = AxisKeys.keyless(AxisKeys.unname(m))
Base.strip(m::NamedDimsArray) = AxisKeys.unname(m)
