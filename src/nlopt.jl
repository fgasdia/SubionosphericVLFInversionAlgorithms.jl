"""
    nlopt_estimate(f, xb; xmin=(65, 0.2), xmax=(90, 1.0), step=(2.0, 0.05),
        method=:LN_COBYLA, neval=600, seed=1234) → (minf, minx, ret)

Perform a nonlinear minimization using `method` given the objective function `f` and initial
estimate of states `xb`.

`f` must be a function of `x` and `grad_x`, even if `grad_x` is not used, e.g. `f(x, grad)`.

`xmin` and `xmax` are two-item tuples for the minimum and maximum ``h′`` and ``β``, respectively. 

`step` is a two-item tuple for the initial step size used for ``h′`` and ``β``, respectively,
by derivative-free methods. The values should be large enough that the value of the
objective function changes significantly, but not too big so that we find the local optimum
nearest to `xb`.
"""
function nlopt_estimate(f, xb; xmin=(65, 0.2), xmax=(90, 1.0), step=(2.0, 0.05),
    method=:LN_COBYLA, neval=600, seed=1234)

    NLopt.srand(seed)

    npts = size(build_xygrid(xb(:h)), 2)

    opt = Opt(method, 2*npts)
    opt.lower_bounds = [fill(xmin[1], npts); fill(xmin[2], npts)]
    opt.upper_bounds = [fill(xmax[1], npts); fill(xmax[2], npts)]
    opt.initial_step = [fill(step[1], npts); fill(step[2], npts)]

    opt.maxeval = neval
    opt.min_objective = f

    x0 = [filter(!isnan, xb(:h)); filter(!isnan, xb(:b))]

    minf, minx, ret = optimize(opt, x0)

    xest = copy(xb)
    xest(:h)[.!isnan.(xb(:h))] .= minx[1:npts]
    xest(:b)[.!isnan.(xb(:b))] .= minx[npts+1:end]

    return minf, xest, ret
end

"""
estimate()

Run the estimation algorithm.
"""
function estimate(neval, oa, op, dt, ρ, ϕ, outdir, scenarioname)
    # Constants
    NLopt.srand(1234)

    h′min, h′max = 60, 92
    βmin, βmax = 0.2, 1.0

    # Estimation steps
    # First, we need to initialize some variables for `for` loop
    let geox, solver

    modelsteps = ((900e3, 1000e3), (600e3, 800e3), (300e3, 600e3), (100e3, 400e3))
    x0 = Float64[]
    for i in eachindex(modelsteps)
        sname = scenarioname*"-$i"





        Δ, r = modelsteps[i]
        mp = ModelParams(100e3, Δ, r, lwrweight(Δ))
        build_model_coords!(MODEL_COORDS, mp)
        Npts = length(MODEL_COORDS)





        ctrlpts = PointSet(model_coords())
        if i == firstindex(modelsteps)
            copy!(x0, [fill(75, Npts); fill(0.4, Npts)])  # daytruth is ~ 73.7, 0.30
        else
            # Interpolate previous minx→x0 to new grid
            problem = EstimationProblem(geox, ctrlpts, (:h′, :β))
            solution = solve(problem, solver)
            copy!(x0, [solution[:h′]; solution[:β]])
        end

        xmin = [fill(h′min, Npts); fill(βmin, Npts)]
        xmax = [fill(h′max, Npts); fill(βmax, Npts)]

        f(x::Vector, grad::Vector) = objective(x, oa, op, dt, mp; ρ, ϕ=m->ϕ(m, mp))

        opt = Opt(:LN_COBYLA, length(x0))
        opt.lower_bounds = xmin
        opt.upper_bounds = xmax
        opt.maxeval = neval

        # For derivative free methods
        # This step size should be big enough that the value of the objective changes significantly,
        # but not too big if you want to find the local optimum nearest to x.
        opt.initial_step = [fill(2.0, Npts); fill(0.05, Npts)]

        opt.min_objective = f

        # Used below for output file
        f0 = f(x0, [])

        @info "Beginning $sname at $(now())"
        minf, minx, ret = optimize(opt, x0)
        @info "Ended $sname at $(now())"

        numevals = opt.numevals # the number of function evaluations
        @info "got $minf at $minx after $numevals iterations (returned $ret)"

        open(joinpath(outdir, sname*".csv"), "a") do io
            # iter, f(x), x...
            println(io, "0,", f0, ",", join(x0, ","))
            println(io, "$numevals,", minf, ",", join(minx, ","))
        end

        # Setup for interpolation of minx → x0 on new coordinates
        geox = georef((h′=minx[1:Npts], β=minx[Npts+1:end]), ctrlpts)
        solver = spatial_solver(mp.w)

        empty!(x0)
        sleep(10)
    end
    end
end

"""
    objective(itp, x, y, paths, dt;
        ρ=l2norm, ϕ=(x)->0, σamp=0.1, σphase=deg2rad(1.0), lwpc=true, pathstep=100e3,
        datatypes::Tuple=(:amp, :phase))

Compute the objective (cost) function given states `x` and observed amplitude and phase in
`y` for the penalty function `ρ(r)` defaulting to the L2-norm on the residuals and an
optional regularization term `ϕ(x)`.

The residuals passed to `ρ` are scaled by `σamp` and `σphase`.

`datatypes` is a tuple that specifies whether `:amp` and/or `:phase` observations should be
used to compute the data error. Even if only one of them is specified.
"""
function objective(itp, x, y, paths, dt;
    ρ=l2norm, ϕ=(x)->0, σamp=0.1, σphase=deg2rad(1.0), lwpc=true, pathstep=100e3,
    datatypes::Tuple=(:amp, :phase))

    ma, mp = model(itp, x, paths, dt; lwpc, pathstep)
    
    if :amp in datatypes && :phase in datatypes
        amp_resids = (y(:amp) .- ma)./σamp
        phase_resids = phasediff.(y(:phase), mp)./σphase
        Δ = [amp_resids; phase_resids]
    elseif :amp in datatypes
        Δ = (y(:amp) .- ma)./σamp
    elseif :phase in datatypes
        Δ = phasediff.(y(:phase), mp)./σphase
    end

    return ρ(Δ) + ϕ(x)
end
