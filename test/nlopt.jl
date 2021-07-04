function nlopt_setup(scenario)
    @unpack dt, modelproj, pathstep, truthfcn, x_grid, y_grid = scenario()

    paths = buildpaths()
    npaths = length(paths)

    # ## Prior states (initial guess)

    # `x` is the prior or "background" estimate of the states. Here we'll initialize an array
    # filled with NaN.
    # Unlike the LETKF, we usually estimate using several different spatial scales. This
    # means we can only hold a single time in each `KeyedArray` `x`.

    dr = 900e3
    lengthscale = 1000e3

    x_grid1 = range(minimum(x_grid)-dr, maximum(x_grid)+dr; step=dr)
    y_grid1 = range(minimum(y_grid)-dr, maximum(y_grid)+dr; step=dr)

    gridshape = (length(y_grid1), length(x_grid1))

    # To compute the localization below, we need a dense grid of lon/lat grid points.
    xy_grid1 = collect(densify(x_grid1, y_grid1))
    lola = permutedims(transform(modelproj, wgs84(), permutedims(xy_grid1)))

    x = KeyedArray(Array{Float64,3}(undef, 2, length(y_grid1), length(x_grid1)),
            field=[:h, :b], y=y_grid1, x=x_grid1)

    x(:h) .= fill(75, gridshape)
    x(:b) .= fill(0.4, gridshape)

    # We also need to define the localization.
    localization = obs2grid_distance(lola, paths; r=lengthscale)
    locmask = anylocal(localization)

    x(:h)[.!locmask] .= NaN
    x(:b)[.!locmask] .= NaN

    # ## Observations

    σA = 0.1  # amplitude, dB
    σp = deg2rad(1.0)  # phase, rad

    # Here we would either load real data or make some fake noisy data using a forward model.
    # We'll do the latter.
    
    oa, op = model(truthfcn, paths, dt; pathstep)  # "truth" observations

    y = KeyedArray(Array{Float64,2}(undef, 2, npaths);
        field=[:amp, :phase], path=pathname.(paths))
    y(:amp) .= oa .+ σA.*randn(npaths)
    y(:phase) .= mod2pi.(op .+ σp.*randn(npaths))

    return paths, x, y, lengthscale
end

function test_nlopt(scenario)
    if Sys.iswindows() && isfile("C:\\LWPCv21\\lwpm.exe")
        @info "    Running NLopt with LWPC. This may take a while."
    else
        @warn "    `test_nlopt()` only runs on Windows with LWPC"
        return
    end

    @unpack pathstep, dt, modelproj = scenario()
    paths, x, y, lengthscale = nlopt_setup(scenario)

    # ## NLOpt-specific arguments

    σamp = 0.1
    σphase = deg2rad(1)

    τ = 2e-7*lengthscale
    solver = LWR(
        :h′ => (weightfun=h->exp(-h^2/(2*τ^2)),),
        :β => (weightfun=h->exp(-h^2/(2*τ^2)),),
        :v => (weightfun=h->exp(-h^2/(2*τ^2)),)
    )

    xygrid = build_xygrid(x(:h))
    itp = GeoStatsInterpolant(solver, modelproj, xygrid)

    # Data and model penalty
    ρ = l2norm

    localizationfcn(lola) = anylocal(obs2grid_distance(lola, paths; r=lengthscale/2))

    μh, μb = 1, 1
    αh, αb = 0, 0
    ϕ1(x) = totalvariation(itp, x, μh, μb, αh, αb; localizationfcn)

    # ## Objective function
    val = objective(itp, x, y, paths, dt;
            ρ, ϕ=ϕ1, σamp, σphase, pathstep, datatypes=(:amp, :phase))

    valwoϕ = objective(itp, x, y, paths, dt;
            ρ, ϕ=(x)->0, σamp, σphase, pathstep, datatypes=(:amp, :phase))
    @test valwoϕ ≈ val  # model cost should be 0 b/c x is flat

    valwoa = objective(itp, x, y, paths, dt;
            ρ, ϕ=(x)->0, σamp, σphase, pathstep, datatypes=(:phase,))
    @test valwoa < val  # data cost no longer includes amplitude penalty

    # ## nlopt_estimate

    μh, μb = 10, 100
    αh, αb = 1.0, 0.1
    ϕ2(x) = totalvariation(itp, x, μh, μb, αh, αb; localizationfcn)

    f(x, grad) = objective(itp, x, y, paths, dt;
        ρ, ϕ=ϕ2, σamp, σphase, pathstep, datatypes=(:amp, :phase))

    minf, xest, ret = nlopt_estimate(f, x;
        xmin=(65, 0.2), xmax=(90, 1.0), step=(2.0, 0.05), neval=50)

    @test minf < f(x, nothing)
end
