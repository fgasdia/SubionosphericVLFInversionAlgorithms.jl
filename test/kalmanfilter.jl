function letkf_setup(scenario)
    @unpack ens_size, ntimes, dt, modelproj, pathstep, x_grid, y_grid, dr,
        lengthscale, truthfcn = scenario()

    paths = buildpaths()
    npaths = length(paths)

    gridshape = (length(y_grid), length(x_grid))
    ncells = prod(gridshape)
    CI = CartesianIndices(gridshape)

    # To compute the covariance relation below, we need the dense matrix of distances from each
    # grid point to every other grid point.
    xy_grid = collect(densify(x_grid, y_grid))
    lola = permutedims(transform(modelproj, wgs84(), permutedims(xy_grid)))

    # ratio of true distance to model distance
    # multiplying by `modelscale` converts a modelproj distance to a WGS84 distance
    truedr = SIA.mediandr(lola)
    modelscale = truedr/dr
    
    # ## Prior states

    # `x` is the prior or "background" estimate of the states. Here we'll initialize an array
    # filled with NaN.

    x = KeyedArray(fill(NaN, 2, length(y_grid), length(x_grid), ens_size, ntimes+1),
            field=[:h, :b], y=y_grid, x=x_grid, ens=1:ens_size, t=0:ntimes)

    # Prior standard deviation of h′ and β
    B = [1.8, 0.04]  # σ_h′, σ_β

    distarr = lonlatgrid_dists(lola)
    gc = gaspari1999_410(distarr, compactlengthscale(lengthscale))
    
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
    replace!(x -> x < 0.16 ? 0.16 : x, b_init)

    # We also need to define the localization. Although it isn't strictly necessary at this
    # point in the setup, the plots look nicer if we mask out the grid cells in the prior
    # ionosphere that will be masked in all the other LETKF update states.
    localization, _ = obs2grid_diamondpill(lola, paths;
        overshoot=sqrt(2)*dr*modelscale, halfwidth=lengthscale/2)
    locmask = anylocal(localization)

    h_init[CI[.!locmask],:] .= NaN
    b_init[CI[.!locmask],:] .= NaN

    x(:h)(t=0) .= h_init
    x(:b)(t=0) .= b_init

    # ## Observations

    # First we define the (digaonal) noise covariance vector `R`.
    # Technically this could vary with every time iteration based on the real data,
    # but usually we assume it is:
    σA = 0.1  # amplitude, dB
    σp = deg2rad(2.0)  # phase, rad

    R = [fill(σA^2, npaths); fill(σp^2, npaths)]

    # Here we would either load real data or make some fake noisy data using a forward model.
    # We'll do the latter.
    
    oa, op = model(truthfcn, paths, dt; pathstep)  # "truth" observations

    y = KeyedArray(Array{Float64,3}(undef, 2, npaths, ntimes);
        field=[:amp, :phase], path=pathname.(paths), t=1:ntimes)
    y(:amp) .= oa .+ σA.*randn(npaths, ntimes)
    y(:phase) .= mod2pi.(op .+ σp.*randn(npaths, ntimes))

    return paths, x, y, R, localization
end

function test_letkf(scenario)
    if Sys.iswindows() && isfile("C:\\LWPCv21\\lwpm.exe")
        @info "    Running LETKF with LWPC. This may take a while (10+ mins)."
    else
        @warn "    `test_letkf()` only runs on Windows with LWPC"
        return
    end

    @unpack pathstep, dt = scenario()
    paths, x, y, R, localization = letkf_setup(scenario)
    npaths = length(paths)
    ntimes = length(y.t)
    ens_size = length(x.ens)
    
    coords = build_xygrid(x(:h)(t=0)(ens=1))
    itp = ScatteredInterpolant(ThinPlate(), esri_102010(), coords)

    # The forward model `H` is usually passed with a wrapper function to return the
    # appropriate KeyedArray.
    # This package provides the function `ensemble_model!` as an example wrapper function
    # for LWPC or LMP.
    ym = KeyedArray(Array{Float64,4}(undef, 2, npaths, ens_size, ntimes+1);
        field=[:amp, :phase], path=pathname.(paths), ens=x.ens, t=0:ntimes)
    H(x,t) = ensemble_model!(ym(t=t), z->model(itp, z, paths, dt; pathstep), x)

    ma, mp = model(itp, x(t=0)(ens=1), paths, dt; pathstep)
    yh = H(x(t=0),0)
    @test yh(ens=1)(:amp) ≈ ma
    @test yh(ens=1)(:phase) ≈ mp

    # Normal LETKF_measupdate
    for i = 1:ntimes
        xa = LETKF_measupdate(x->H(x,i-1), x(t=i-1), y(t=i), R;
            ρ=1.1, localization, datatypes=(:amp, :phase))
        @test all(x->abs(x)<120, ym(t=i-1)(:amp))
        @test all(x->abs(x)<4π, ym(t=i-1)(:phase))
        x(t=i) .= xa
        sleep(0.5)
    end

    # LETKF_measupdate w/ amplitude only
    for i = 1:ntimes
        xa = LETKF_measupdate(x->H(x,i-1), x(t=i-1), y(t=i), R[1:npaths];
            ρ=1.1, localization, datatypes=(:amp,))
        @test all(x->abs(x)<120, ym(t=i-1)(:amp))
        x(t=i) .= xa
        sleep(0.5)
    end

    # LETKF_measupdate w/ phase only
    for i = 1:ntimes
        xa = LETKF_measupdate(x->H(x,i-1), x(t=i-1), y(t=i), R[npaths+1:end];
            ρ=1.1, localization, datatypes=(:phase,))
        @test all(x->abs(x)<4π, ym(t=i-1)(:phase))
        x(t=i) .= xa
        sleep(0.5)
    end
end
