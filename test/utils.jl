function testscenario(dr=500e3)
    dt = DateTime(2019, 2, 15, 18, 30)
    tx = TRANSMITTER[:NAA]
    rx = Receiver("Boulder", 40.02, -105.27, 0.0, VerticalDipole())
    paths = [(tx, rx)]

    westbound, eastbound = -109.5, -63
    southbound, northbound = 36.5, 48.3

    bounds = [westbound northbound; eastbound northbound; westbound southbound; eastbound southbound]
    pts = transform(wgs84(), esri_102010(), bounds)
    xmin, xmax = extrema(pts[:,1])
    ymin, ymax = extrema(pts[:,2])

    x_grid = range(xmin, xmax; step=dr)
    y_grid = range(ymin, ymax; step=dr)

    x = KeyedArray(fill(NaN, 2, length(y_grid), length(x_grid)); field=[:h, :b], y=y_grid, x=x_grid)
    x(:h) .= 70.0
    x(:b) .= 0.4

    return x, paths, dt
end

Base.strip(m::KeyedArray) = AxisKeys.keyless(AxisKeys.unname(m))
