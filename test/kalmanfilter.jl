
function tmppaths()
    transmitters = [TRANSMITTER[:NLK], TRANSMITTER[:NML]]

    receivers = [
        Receiver("Whitehorse", 60.724, -135.043, 0.0, VerticalDipole()),
        # Receiver("Churchill", 58.74, -94.085, 0.0, VerticalDipole()),
        # Receiver("Stony Rapids", 59.253, -105.834, 0.0, VerticalDipole()),
        # Receiver("Fort Smith", 60.006, -111.92, 0.0, VerticalDipole()),
        # Receiver("Bella Bella", 52.1675508, -128.1545219, 0.0, VerticalDipole()),
        # Receiver("Nahanni Butte", 61.0304412, -123.3926734, 0.0, VerticalDipole()),
        # Receiver("Juneau", 58.32, -134.41, 0.0, VerticalDipole()),
        # Receiver("Ketchikan", 55.35, -131.673, 0.0, VerticalDipole()),
        # Receiver("Winnipeg", 49.8822, -97.1308, 0.0, VerticalDipole()),
        # Receiver("IslandLake", 53.8626, -94.6658, 0.0, VerticalDipole()),
        # Receiver("Gillam", 56.3477, -94.7093, 0.0, VerticalDipole())
    ]

    return [(tx, rx) for tx in transmitters for rx in receivers]
end

"""
    arguments()

Toy problem that demonstrates the correct form of the arguments to `LETKF_measupdate`.
"""
function arguments()
    # ## Setup

    # Later we will need the propagation paths. Here they are:
    transmitters = [TRANSMITTER[:NLK], TRANSMITTER[:NML]]
    receivers = [
        Receiver("Churchill", 58.74, -94.085, 0.0, VerticalDipole()),
        Receiver("Stony Rapids", 59.253, -105.834, 0.0, VerticalDipole()),
        Receiver("Fort Smith", 60.006, -111.92, 0.0, VerticalDipole()),
    ]

    paths = [(tx, rx) for tx in transmitters for rx in receivers]
    npaths = length(paths)

    # y_grid and x_grid (or `y` and `x`) are the coordinates of estimation grid points in
    # the y-axis and x-axis. These could be latitude and longitude, respectively, but it
    # is better to use a plane projection so that the grid points are more equally spaced.
    y_grid = 45:70
    x_grid = -130:-90

    gridshape = (length(y_grid), length(x_grid))  # useful later
    CI = CartesianIndices(gridshape)

    # The DateTime will be needed for the prior ionosphere and possibly the IGRF magnetic field
    dt = DateTime(2019, 2, 15, 18, 30)

    ens_size = 100  # size of the ensemble... the number of ionospheres
    ntimes = 6  # how many time steps to take
    
    # ## Prior states

    # `xb` is the prior or "background" estimate of `x`. Here we'll begin with an array
    # filled with NaN.
    xb = KeyedArray(fill(NaN, 2, length(y_grid), length(x_grid), ens_size, ntimes+1),
            field=[:h, :b], y=y_grid, x=x_grid, ens=1:ens_size, t=0:ntimes)

    # The filter wouldn't do anything if `xb` were all NaN, so let's make up some initial
    # estimate of the ionosphere and fill it in to `t = 0`.
    # 
    # We'll use the Ferguson ionosphere model (which isn't very good at any time other than
    # noon or midnight, but...) as the mean of the initial ensemble and the Gaussian-like
    # covariance relation from Gaspari & Cohn 1999 (doi: 10.1002/qj.49712555417) to
    # describe the spatial correlation length of the ionosphere.

    # Prior variance of h′ and β
    Bstd = [1.8, 0.04]  # σ_h′, σ_β
    B = Bstd.^2

    # To compute the covariance relation, we need the dense matrix of distances from each
    # grid point to every other grid point.
    lonlats = [(x, y) for x in x_grid for y in y_grid]
    lonlats = permutedims([getindex.(lonlats, 1) getindex.(lonlats, 2)])

    distarr = lonlatgrid_dists(lonlats)
    gc = gaspari1999_410(distarr, 2000e3)  # scale length of 2000 km

    # The mean ionosphere is the Ferguson ionosphere model
    prior = [ferguson(lonlats[2,i], zenithangle(lonlats[2,i], lonlats[1,i], dt), dt) for i in axes(lonlats,2)]
    h_prior, b_prior = getindex.(prior, 1), getindex.(prior, 2)

    # We can sample 100 ionospheres from this multivariate-Gaussian distribution using
    # Distributions.jl.
    # First we define the distribution itself.
    hdistribution = MvNormal(h_prior, B[1]*gc)
    bdistribution = MvNormal(b_prior, B[2]*gc)

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
    localization, _ = obs2grid_diamondpill(lonlats, paths; overshoot=200e3, halfwidth=600e3)
    for i in axes(localization,1)
        # Check if not a single path affects gridcell i
        if all(x->x==0, localization[i,:])
            h_init[CI[i],:] .= NaN
            b_init[CI[i],:] .= NaN
        end
    end

    # Finally, let's fill in the `xb` table
    xb(:h)(t=0) .= h_init
    xb(:b)(t=0) .= b_init

    # ## Simulated "true" observations

    # Here we would either load real data or make some fake noisy data using a forward model.
    # In this example we will fake it completely so you don't need to wait for a real
    # forward model.

    # First we define the (digaonal) noise covariance vector `R`.
    # Technically this could vary with every time iteration based on the real data,
    # but usually we assume it is:
    σA = 0.1  # amplitude, dB
    σp = deg2rad(1.0)  # phase, rad

    R = [fill(σA^2, npaths); fill(σp^2, npaths)]

    # Let's make a KeyedArray for the data `y`
    y = KeyedArray(Array{Float64,3}(undef, 2, npaths, ntimes);
        field=[:amp, :phase], path=pathname.(paths), t=1:ntimes)
    y(:amp) .= 100 .+ σA.*rand(npaths, ntimes)
    y(:phase) .= mod2pi.(σp.*rand(npaths, ntimes))

    # The forward model `H` is usually passed with a wrapper function to return the
    # appropriate KeyedArray.
    # This package provides the function `ensemble_model` as an example wrapper function
    # for LWPC or LMP.

    # Usually H would look something like this:
    #==
    model_projection = esri_102010
    lwpc = true
    itp = ScatteredInterpolant(ThinPlate(), model_projection)
    H(x) = ensemble_model(model(itp, x, paths, dt; pathstep=100e3, lwpc), x, pathnames; deg=lwpc)
    ==#
    
    # But here we'll use a fake H
    H(x) = ensemble_model(x->(rand(length(paths)), mod2pi.(rand(length(paths)))), x, pathname.(paths); deg=false)

    # First, let's create a KeyedArray to save the forward model run results to.
    ym = KeyedArray(Array{Float64,4}(undef, 2, npaths, ens_size, ntimes);
        field=[:amp, :phase], path=pathname.(paths), ens=xb.ens, t=0:ntimes-1)

    # Let's use time t = 1
    i = 1
    LETKF_measupdate(H, xb(t=i-1), y(t=i), R;
        ρ=1.1, localization=localization, datatypes=(:amp, :phase), deg=false)
end
