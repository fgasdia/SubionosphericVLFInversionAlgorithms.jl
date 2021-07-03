function test_grids()
    x, paths, _ = testscenario(500e3)

    # From testscenario
    west, east = -109.5, -63
    south, north = 36.5, 48.3
    
    x_grid, y_grid = build_xygrid(west, east, south, north, wgs84(), esri_102010(); dr=500e3)
    @test x_grid isa AbstractRange
    @test y_grid isa AbstractRange
    @test x_grid == x.x && y_grid == x.y  # to make sure bounds match

    xy_grid = densify(x_grid, y_grid)
    @test size(xy_grid) == (2, length(x_grid)*length(y_grid))

    @test xy_grid == build_xygrid(x)

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
        :h′ => (weightfun=f,),
        :β => (weightfun=f,)
    )
    itp = GeoStatsInterpolant(solver, esri_102010(), xy_grid)
    hgrid = dense_grid(itp, x(:h), x_grid, y_grid)
    @test all(isapprox(70), hgrid)

    fine_hgrid = dense_grid(itp, x(:h), fine_xgrid, fine_ygrid)
    @test all(isapprox(70; atol=0.1), fine_hgrid[1:2:end])     

    # Localization

    lola = permutedims(transform(esri_102010(), wgs84(), permutedims(xy_grid)))
    distarr = lonlatgrid_dists(lola)
    @test size(distarr) == (size(lola,2), size(lola,2))
    @test iszero(diag(distarr))

    c = 2000e3
    gc = gaspari1999_410(distarr, c)
    @test size(gc) == size(distarr)
    @test all(isequal(1), diag(gc))

    @test compactlengthscale(gaussianstddev(c)) ≈ c

    localization, _ = obs2grid_diamondpill(lola, paths)
    @test size(localization) == (size(lola,2), length(paths))
    # heatmap(x_grid, y_grid, reshape(localization, (length(y_grid), length(x_grid))))

    gcp_boundary, _ = SIA.boundary_coords(paths)
    @test gcp_boundary isa Vector  # sanity

    localization2 = obs2grid_distance(lola, paths)
    @test size(localization2) == size(localization)
    # heatmap(x_grid, y_grid, reshape(localization2, (length(y_grid), length(x_grid))))
end
