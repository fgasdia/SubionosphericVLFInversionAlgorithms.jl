function dayscenario()
    # ## Setup

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
    npaths = length(paths)

    # The DateTime will be needed for the prior ionosphere and possibly the IGRF magnetic field
    dt = DateTime(2020, 3, 1, 20, 00)  # day

    ens_size = 30  # size of the ensemble... the number of ionospheres
    ntimes = 2  # how many time steps to take

    # y_grid and x_grid (or `y` and `x`) are the coordinates of estimation grid points in
    # the y-axis and x-axis. These could be latitude and longitude, respectively, but it
    # is better to use a plane projection so that the grid points are more equally spaced.
    westbound, eastbound = -136, -90
    southbound, northbound = 49, 62
    dr = 500e3  # m

    bounds = [westbound northbound; eastbound northbound; westbound southbound; eastbound southbound]
    pts = transform(wgs84(), esri_102010(), bounds)
    (xmin, xmax), (ymin, ymax) = extrema(pts, dims=1)

    x_grid = range(xmin, xmax; step=dr)
    y_grid = range(ymin, ymax; step=dr)

    gridshape = (length(y_grid), length(x_grid))  # useful later
    ncells = prod(gridshape)
    CI = CartesianIndices(gridshape)

    # To compute the covariance relation below, we need the dense matrix of distances from each
    # grid point to every other grid point.
    xy_grid = Matrix{Float64}(undef, 2, ncells)
    for i in axes(xy_grid,2)
        xy_grid[:,i] .= (x_grid[CI[i][2]], y_grid[CI[i][1]])
    end
    lola = permutedims(transform(esri_102010(), wgs84(), permutedims(xy_grid)))
    
    # ## Prior states

    # `xb` is the prior or "background" estimate of `x`. Here we'll begin with an array
    # filled with NaN.

    x = KeyedArray(fill(NaN, 2, length(y_grid), length(x_grid), ens_size, ntimes+1),
            field=[:h, :b], y=y_grid, x=x_grid, ens=1:ens_size, t=0:ntimes)

    # Prior standard deviation of h′ and β
    B = [1.8, 0.04]  # σ_h′, σ_β

    distarr = lonlatgrid_dists(lola)
    gc = gaspari1999_410(distarr, 2000e3)  # scale length of 2000 km
    
    # We can sample 100 ionospheres from this multivariate-Gaussian distribution using
    # Distributions.jl.
    # First we define the distribution itself.
    hdistribution = MvNormal(fill(75, ncells), B[1]^2*gc)  # yes, w/ matrix argument we need variance
    bdistribution = MvNormal(fill(0.4, ncells), B[2]^2*gc)

    # Then we build the ensemble and reshape it to the grid shape
    h_init = reshape(rand(hdistribution, ens_size), gridshape..., ens_size)
    b_init = reshape(rand(bdistribution, ens_size), gridshape..., ens_size)

    # LWPC has issues with low β ionospheres. Lets clip the bottom of the β distribution
    # just in case. This isn't strictly necessary if using LongwaveModePropagator.jl, but
    # these low β ionospheres are fairly unrealistic anyways
    replace!(x->x < 0.16 ? 0.16 : x, b_init)

    # We also need to define the localization. Although it isn't strictly necessary at this
    # point in the setup, the plots look nicer if we mask out the grid cells in the prior
    # ionosphere that will be masked in all the other LETKF update states.
    localization, _ = obs2grid_diamondpill(lola, paths; overshoot=200e3, halfwidth=600e3)
    for i in axes(localization,1)
        # Check if not a single path affects gridcell i
        if all(x->x==0, localization[i,:])
            h_init[CI[i],:] .= NaN
            b_init[CI[i],:] .= NaN
        end
    end
    x(:h)(t=0) .= h_init
    x(:b)(t=0) .= b_init

    # ## Observations

    # First we define the (digaonal) noise covariance vector `R`.
    # Technically this could vary with every time iteration based on the real data,
    # but usually we assume it is:
    σA = 0.1  # amplitude, dB
    σp = deg2rad(1.0)  # phase, rad

    R = [fill(σA^2, npaths); fill(σp^2, npaths)]

    # Here we would either load real data or make some fake noisy data using a forward model.
    # We'll do the latter.

    # trueiono = [ferguson(lola[2,i], zenithangle(lola[2,i], lola[1,i], dt), dt) for i in axes(lola,2)]
    hbfcn(lo, la, dt) = ferguson(la, zenithangle(la, lo, dt), dt)
    oa, op = model(hbfcn, paths, dt; pathstep=500e3)  # "truth" observations

    y = KeyedArray(Array{Float64,3}(undef, 2, npaths, ntimes);
        field=[:amp, :phase], path=pathname.(paths), t=1:ntimes)
    y(:amp) .= oa .+ σA.*randn(npaths, ntimes)
    y(:phase) .= mod2pi.(op .+ σp.*rand(npaths, ntimes))

    return paths, x, y, R, localization, dt
end

function test_letkf(scenario)
    if Sys.iswindows() && isfile("C:\\LWPCv21\\lwpm.exe")
        @info " Running LETKF with LWPC. This may take a while."
    else
        @warn "Kalman filter: `test_day()` only runs on Windows with LWPC"
        return
    end

    paths, x, y, R, localization, dt = scenario()
    npaths = length(paths)
    ntimes = length(y.t)
    ens_size = length(x.ens)
    
    itp = ScatteredInterpolant(ThinPlate(), esri_102010())

    # The forward model `H` is usually passed with a wrapper function to return the
    # appropriate KeyedArray.
    # This package provides the function `ensemble_model!` as an example wrapper function
    # for LWPC or LMP.
    ym = KeyedArray(Array{Float64,4}(undef, 2, npaths, ens_size, ntimes+1);
        field=[:amp, :phase], path=pathname.(paths), ens=x.ens, t=0:ntimes)
    H(x,t) = SIA.ensemble_model!(ym(t=t), z->model(itp, z, paths, dt; pathstep=500e3, lwpc=true), x)

    ma, mp = model(itp, x(t=0)(ens=1), paths, dt; pathstep=500e3, lwpc=true)
    yh = H(x(t=0),0)
    @test yh(ens=1)(:amp) ≈ ma
    @test yh(ens=1)(:phase) ≈ mp

    # Normal LETKF_measupdate
    for i = 1:ntimes
        xa = LETKF_measupdate(x->H(x,i-1), x(t=i-1), y(t=i), R;
            ρ=1.1, localization=localization, datatypes=(:amp, :phase))
        @test !isnothing(xa)
        @test all(x->abs(x)<120, ym(t=i-1)(:amp))
        @test all(x->abs(x)<4π, ym(t=i-1)(:phase))
        x(t=i) .= xa
    end

    # LETKF_measupdate w/ amplitude only
    for i = 1:ntimes
        xa = LETKF_measupdate(x->H(x,i-1), x(t=i-1), y(t=i), R[1:npaths];
            ρ=1.1, localization=localization, datatypes=(:amp,))
        @test !isnothing(xa)
        @test all(x->abs(x)<120, ym(t=i-1)(:amp))
        x(t=i) .= xa
    end

    # LETKF_measupdate w/ phase only
    for i = 1:ntimes
        xa = LETKF_measupdate(x->H(x,i-1), x(t=i-1), y(t=i), R[npaths+1:end];
            ρ=1.1, localization=localization, datatypes=(:phase,))
        @test !isnothing(xa)
        @test all(x->abs(x)<4π, ym(t=i-1)(:phase))
        x(t=i) .= xa
    end
end
