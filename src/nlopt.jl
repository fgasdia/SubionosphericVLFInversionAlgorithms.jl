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
    method=:LN_COBYLA, neval=600, seed=1234, use_local_opt=false)

    NLopt.srand(seed)

    npts = size(build_xygrid(xb(:h)), 2)

    opt = Opt(method, 2*npts)
    opt.lower_bounds = [fill(xmin[1], npts); fill(xmin[2], npts)]
    opt.upper_bounds = [fill(xmax[1], npts); fill(xmax[2], npts)]
    opt.initial_step = [fill(step[1], npts); fill(step[2], npts)]

    opt.maxeval = neval
    opt.min_objective = f

    # TEMP? (to ensure we stop only based on maxeval)
    # opt.ftol_rel = 1e-16
    # opt.ftol_abs = 1e-16
    # opt.xtol_rel = 1e-16
    # opt.xtol_abs = 1e-16

    if use_local_opt
        local_opt = Opt(:LN_COBYLA, 2*npts)
        local_opt.initial_step = [fill(step[1], npts); fill(step[2], npts)]
        local_opt.maxeval = neval ÷ 10

        opt.local_optimizer = local_opt
    end

    x0 = [filter(!isnan, xb(:h)); filter(!isnan, xb(:b))]

    minf, minx, ret = optimize(opt, x0)

    xest = copy(xb)
    # gridshape = (length(xest.y), length(xest.x))
    xest(:h)[.!isnan.(xb(:h))] .= minx[1:npts]
    xest(:b)[.!isnan.(xb(:b))] .= minx[npts+1:end]

    return minf, xest, ret, opt
end

"""
    objective(itp, x, y, paths, dt;
        ρ=l2norm, ϕ=(x)->0, σamp=0.1, σphase=deg2rad(1.0), lwpc=true, pathstep=100e3,
        numexe=16, datatypes::Tuple=(:amp, :phase))

Compute the objective (cost) function given states `x` and observed amplitude and phase in
`y` for the penalty function `ρ(r)` defaulting to the L2-norm on the residuals and an
optional regularization term `ϕ(x)`.

The residuals passed to `ρ` are scaled by `σamp` and `σphase`.

`datatypes` is a tuple that specifies whether `:amp` and/or `:phase` observations should be
used to compute the data error. Even if only one of them is specified.
"""
function objective(itp, x, y, paths, dt;
    ρ=l2norm, ϕ=(x)->0, σamp=0.1, σphase=deg2rad(1.0), lwpc=true, pathstep=100e3, numexe=16,
    datatypes::Tuple=(:amp, :phase))

    ma, mp = model(itp, x, paths, dt; lwpc, pathstep, numexe)
    
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
