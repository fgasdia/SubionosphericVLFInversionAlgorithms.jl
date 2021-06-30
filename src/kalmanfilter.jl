"""
LETKF_measupdate(H, xb, y, R, y_grid, x_grid, pathnames, ens_size;
    ρ=1.1, localization=nothing, datatypes=(:amp, :phase)) → (xa, yb)

LETKF (Local Ensemble Transform Kalman Filter) analysis update applied locally, following
the steps in [^1].

# Arguments

This function is specific to the VLF estimation problem and makes use of `KeyedArray`s from
AxisKeys.jl.

- `H → KeyedArray(yb; field=[:amp, :phase], path=pathnames, ens=ens)`:
    Observation model that maps from state space to observation space (``y = H(x) + ϵ``).
- `xb::KeyedArray(xb; field=[:h, :b], y=y, x=x,  ens=ens)`:
    Ensemble matrix of states having size `(nstates, nensemble)`.
    It is assumed the first half of rows are ``h′`` and the second half are ``β``.
- `y::KeyedArray(data; field=[:amp, :phase], path=pathnames)`:
    Stacked vector of observations `[amps...; phases...]`.
- `R`: Vector of the diagonal data covariance matrix ``σ²``.
- `y_grid`: 

# References

[^1]: B. R. Hunt, E. J. Kostelich, and I. Szunyogh, “Efficient data assimilation for
spatiotemporal chaos: A local ensemble transform Kalman filter,” Physica D: Nonlinear
Phenomena, vol. 230, no. 1, pp. 112–126, Jun. 2007.
"""
function LETKF_measupdate(H, xb, y, R;
    ρ=1.1, localization=nothing, datatypes::Tuple=(:amp, :phase))

    # Make sure xb, yb, and y are correct KeyedArrays
    # xb = KeyedArray(xb; field=[:h, :b], y=xb.y, x=xb.x, ens=xb.ens)
    # y = KeyedArray(y; field=[:amp, :phase], path=y.path)

    gridshape = (length(xb.y), length(xb.x))
    ncells = prod(gridshape)
    npaths = length(y.path)
    ens_size = length(xb.ens)

    if !isnothing(localization)
        size(localization) == (ncells, npaths) ||
            throw(ArgumentError("Size of `localization` must be `(ncells, npaths)`"))
    end

    # 1.
    yb = H(xb)
    # yb = KeyedArray(yb; field=[:amp, :phase], path=y.path, ens=xb.ens)
    
    ybar = mean(yb, dims=:ens)

    if :amp in datatypes && :phase in datatypes
        Y = similar(yb)
        Y(:amp) .= yb(:amp) .- ybar(:amp)
        Y(:phase) .= phasediff.(yb(:phase), ybar(:phase))
    elseif :amp in datatypes
        Y = yb(:amp) .- ybar(:amp)
    elseif :phase in datatypes
        Y = phasediff.(yb(:phase), ybar(:phase))
    end

    # 2.
    xbbar = mean(xb, dims=:ens)
    Xb = xb .- xbbar

    # 3. Localization
    xa = similar(xb)
    CI = CartesianIndices(gridshape)
    for n in 1:ncells
        yidx, xidx = CI[n][1], CI[n][2]

        # Currently localization is binary (cell is included or not)
        if isnothing(localization)
            loc_mask = trues(npaths)
        else
            loc = view(localization, n, :)
            loc_mask = loc .> 0
            if !any(loc_mask)
                # No measurements in range, nothing to update
                xa(y=Index(yidx), x=Index(xidx)) .= xb(y=Index(yidx), x=Index(xidx))
                continue
            end
        end

        # Localize and flatten measurements
        ybar_loc = ybar(path=Index(loc_mask))
        Y_loc = Y(path=Index(loc_mask))
        y_loc = y(path=Index(loc_mask))

        if :amp in datatypes && :phase in datatypes
            Y_loc = [Y_loc(:amp); Y_loc(:phase)]
            R_loc = @views Diagonal([R[1:npaths][loc_mask]; R[npaths+1:end][loc_mask]])
        else
            # Only amp or phase
            R_loc = @views Diagonal(R[loc_mask])
        end

        # 4.
        C = strip(Y_loc)'/R_loc

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv((ens_size - 1)*I/ρ + C*Y_loc)

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(strip(Patilde)))

        # 7.
        if :amp in datatypes && :phase in datatypes
            Δ = [y_loc(:amp) .-  ybar_loc(:amp); phasediff.(y_loc(:phase), ybar_loc(:phase))]
        elseif :amp in datatypes
            Δ = y_loc(:amp) .- ybar_loc(:amp)
        elseif :phase in datatypes
            Δ = phasediff.(y_loc(:phase), ybar_loc(:phase))
        end

        wabar = Patilde*C*Δ
        wa = Wa .+ wabar

        # 8.
        xbbar_loc = xbbar(y=Index(yidx), x=Index(xidx))
        Xb_loc = Xb(y=Index(yidx), x=Index(xidx))

        xa(y=Index(yidx), x=Index(xidx)) .= Xb_loc*wa .+ xbbar_loc
    end

    return xa
end

"""
    ensemble_model!(ym, f, x)

Run the forward model `f` with `KeyedArray` argument `x` for each member of `x.ens`.
"""
function ensemble_model!(ym, f, x)
    # ym = KeyedArray(Array{Float64,3}(undef, 2, length(pathnames), length(x.ens));
    #         field=SVector(:amp, :phase), path=pathnames, ens=x.ens)
    for e in x.ens
        a, p = f(x(ens=e))
        ym(:amp)(ens=e) .= a
        ym(:phase)(ens=e) .= p
    end

    # Fit a Gaussian to phase data ensemble, then use wrap the phases from ±180° from the mean
    for p in ym.path
        ym(:phase)(path=p) .= modgaussian(ym(:phase)(path=p))
    end

    return ym
end

"""
    modgaussian(phases)

Fit a Gaussian distribution to a vector of `phases` in radians and return the phases shifted
such they are wrapped within ±π about the mean of the fit.
"""
function modgaussian(phases)
    μ = fit(Normal{Float64}, phases).μ
    return mod2pi.(phases .- μ .+ π) .+ μ .- π
end
