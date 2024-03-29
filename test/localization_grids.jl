function test_grids()
    x, paths, _ = testscenario(500e3)

    # From testscenario
    west, east = -109.5, -63
    south, north = 36.5, 48.3
    
    dr = 500e3
    x_grid, y_grid = build_xygrid(west, east, south, north, wgs84(), esri_102010(); dr)
    @test x_grid isa AbstractRange
    @test y_grid isa AbstractRange
    @test x_grid == x.x && y_grid == x.y  # to make sure bounds match

    xy_grid = densify(x_grid, y_grid)
    @test size(xy_grid) == (2, length(x_grid)*length(y_grid))

    @test xy_grid == build_xygrid(x(:h))

    itp = ScatteredInterpolant(ThinPlate(), esri_102010(), xy_grid)
    hgrid = dense_grid(itp, x(:h), x_grid, y_grid)
    @test all(isapprox(70), hgrid)

    fine_xgrid = range(first(x_grid), last(x_grid); step=step(x_grid)/2)
    fine_ygrid = range(first(y_grid), last(y_grid); step=step(y_grid)/2)

    fine_hgrid = dense_grid(itp, x(:h), fine_xgrid, fine_ygrid)
    @test all(isapprox(70; atol=0.1), fine_hgrid[1:2:end])

    τ = 2e-7*500e3
    f(h) = exp(-h^2/(2*τ^2))
    solver = LWR(
        :h => (weightfun=f,),
        :b => (weightfun=f,),
        :v => (weightfun=f,)
    )
    itp = GeoStatsInterpolant(solver, esri_102010(), xy_grid)
    hgrid = dense_grid(itp, x(:h), x_grid, y_grid)
    @test all(isapprox(70), hgrid)

    fine_hgrid = dense_grid(itp, x(:h), fine_xgrid, fine_ygrid)
    @test all(isapprox(70; atol=0.1), fine_hgrid[1:2:end])     

    # Localization

    trans = Proj.Transformation(esri_102010(), wgs84())
    lola = trans.(parent(parent(xy_grid)))  # parent undoes the reshape reinterpret to get a vector of tuples
    distarr = lonlatgrid_dists(lola)
    @test size(distarr) == (length(lola), length(lola))
    @test iszero(diag(distarr))

    mdr = mediandr(lola)
    @test dr != mdr
    @test abs(dr - mdr) < 1e5

    c = 2000e3
    gc = gaspari1999_410(distarr, c)
    @test size(gc) == size(distarr)
    @test all(isequal(1), diag(gc))

    @test compactlengthscale(gaussianstddev(c)) ≈ c

    localization, _ = obs2grid_diamondpill(lola, paths)
    @test size(localization) == (length(lola), length(paths))
    # heatmap(x_grid, y_grid, reshape(localization, (length(y_grid), length(x_grid))))

    locmask = anylocal(localization)
    x = KeyedArray(fill(1.0, length(y_grid), length(x_grid)); y=y_grid, x=x_grid)
    x[.!locmask] .= NaN
    @test build_xygrid(locmask, x_grid, y_grid) == build_xygrid(x)

    gcp_boundary, _ = SIA.boundary_coords(paths)
    @test gcp_boundary isa Vector  # sanity

    localization2 = obs2grid_distance(lola, paths)
    @test size(localization2) == size(localization)
    # heatmap(x_grid, y_grid, reshape(localization2, (length(y_grid), length(x_grid))))

    locdistances = obs2grid_distances(lola, paths)
    @test size(locdistances) == size(localization)
end
