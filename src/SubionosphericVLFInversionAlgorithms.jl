module SubionosphericVLFInversionAlgorithms

using Core: Argument
using Random, Statistics, LinearAlgebra
using StaticArrays, AxisKeys
using ProgressMeter

include("utils.jl")

export vfsa, LETKF_measupdate
export gaspari_cohn99_410

const RNG = MersenneTwister(1234)


"""
    vfsa(f, x, xmin, xmax, Ta, Tm; NT=1, NK=1000, Ta_min=typemin(Ta(1)), E_min=nothing,
        saveprogress=:false, filename=nothing, rng=RNG)

Apply very fast simulated annealing (VFSA) to the function `f`, returning a tuple of the
best `x` and corresponding energy `E = f(x)`.

This function will run for `NK` total iterations or until: `Ta < Ta_min` or `E < E_min`.
By default `Ta_min` and `E_min` are effectively ignored.
The progress meter will only reflect progress based on `NT*NK`, but the temperature and
error will be printed in the meter.

# Arguments

- `f`: objective function (to minimize) of `x`. Must return a scalar value.
- `x`: initial parameter (model) values.
- `xmin`: lower bound of each model parameter.
- `xmax`: upper bound of each model parameter.
- `Ta`: temperature function of iteration `k` for acceptance criterion.
- `Tm`: temperature function of iteration `k`, indexable for each model parameter.
- `NT=1`: number of moves at each temperature.
- `NK=1000`: total number of iterations (calls of `f`).
- `Ta_min=typemin(Ta(1))`: minimum temperature `Ta` before returning.
- `E_min=nothing`: minimum error `f(x)` before returning.
- `saveprogress=:false`:
    If `saveprogress` is `:false`, return `(x, E)`. If `:all`, return
    `(x, E, xprogress, Eprogress)` where the "progress" variables save every intermediate
    value of `x` and `E`.
- `filename=nothing`: if not `nothing`, save `[iteration Ta E x...]` to a CSV file `filename`
    based on argument of `saveprogress`. Because this appends to the file at each iteration,
    it is much slower than saving `xprogress` in one step. `filename` should only be set for
    expensive objective functions `f`.
- `rng=RNG`: optional random number generator. Otherwise, a package-level `MersenneTwister`
    RNG will be used.

!!! note
    
    For VFSA, the temperature function should be of the form `T(k) = T0*exp(-c*k^(1/NM))`
    where `NM` is the number of model parameters `length(x)`. `Tm` can return multuple
    values for each of `x`.

# References

[1]: Kochenderfer, Mykel J., and Tim A. Wheeler. Algorithms for Optimization. The MIT Press,
    2019, https://algorithmsbook.com/optimization/#.

[2]: Sen, Mrinal K., and Paul L. Stoffa. “Ch. 4: Simulated Annealing Methods.” Global
    Optimization Methods in Geophysical Inversion, 2nd ed., Cambridge University Press,
    2013, doi:10.1017/CBO9780511997570.
"""
function vfsa(f, x, xmin, xmax, Ta, Tm; NT=1, NK=1000, Ta_min=typemin(Ta(1)), E_min=nothing,
        saveprogress=:false, filename=nothing, rng=RNG)
    length(x) == length(xmin) == length(xmax) ||
        throw(ArgumentError("`x`, `xmin`, and `xmax` must have same length"))
    all(xmin .< xmax) || throw(ArgumentError("`xmin` must be less than `xmax`"))

    # Initialize
    NM = length(x)
    x = copy(x)  # so `x` isn't modified in place
    x′ = similar(x)
    
    E = f(x)
    if isnothing(E_min)
        E_min = typemin(E)
    end
    early_exit = false  # flag if we exit earlier than NK*NT

    if saveprogress == :false
        xprogress = nothing
        Eprogress = nothing
    elseif saveprogress == :all
        xprogress = Matrix{eltype(x)}(undef, NM, NK*NT)
        Eprogress = Vector{typeof(E)}(undef, NK*NT)

        if !isnothing(filename)
            let E=E, x=x  # otherwise Core.Box type-instability w/ do
                open(filename, "a") do io
                    println(io, 0, ",", Ta(0.0), ",", E, ",", join(x, ","))
                end
            end
        end
    end

    pm = Progress(NK*NT)
    generate_showvalues(iter, T, E) = () -> [(:iter,iter), (:Ta,T), (:E,E)]
    
    iter = 1
    for k in 1:NK
        Ta_k = Ta(k)
        Tm_k(i) = (t = Tm(k); t isa Number ? t : t[i])

        if Ta_k < Ta_min || E <= E_min
            iter -= 1  # undo the increment from end of last loop
            early_exit = true
            finish!(pm)
            break
        end

        for n in 1:NT
            for i in 1:NM
                xnew = typemax(x[i])
                while !(xmin[i] <= xnew <= xmax[i])
                    # keep trying until new estimate is in bounds (usually called only once)
                    u = rand(rng)
                    y = sign(u - 1/2)*Tm_k(i)*((1 + 1/Tm_k(i))^abs(2*u - 1) - 1)
                    xnew = x[i] + y*(xmax[i] - xmin[i])
                end
                x′[i] = xnew
            end

            E′ = f(x′)
            ΔE = E′ - E

            if ΔE <= 0 || rand(rng) < exp(-ΔE/Ta_k)
                x .= x′
                E = E′
            end

            if saveprogress == :all
                xprogress[:,iter] .= x
                Eprogress[iter] = E

                if !isnothing(filename)
                    let iter=iter, E=E, x=x  # otherwise Core.Box type-instability w/ do
                        open(filename, "a") do io
                            println(io, iter, ",", Ta_k, ",", E, ",", join(x, ","))
                        end
                    end
                end
            end
            iter += 1
            next!(pm; showvalues=generate_showvalues(iter, Ta_k, E))
        end
    end

    if saveprogress != :false
        if early_exit
            # Trim unused rows (because of early exit)
            resize!(Eprogress, iter)
            xprogress = xprogress[:,1:iter]
        end

        # It's faster to write to xprogress as (NM, NK*NT) above, but more useful to have
        # (NK*NT, NM) for processing
        xprogress = permutedims(xprogress)
    end

    return x, E, xprogress, Eprogress
end

"""
    LETKF_measupdate(f, xb, data, R, y_grid, x_grid, pathnames, ens_size;
        ρ=1.1, localization=nothing, datatypes=(:amp, :phase)) → (xa, yb)

LETKF (Local Ensemble Transform Kalman Filter) analysis update applied locally, following
the steps in [^1].

# Arguments

- `xb::AbstractMatrix`: Ensemble matrix of states having size `(nstates, nensemble)`.
    Note!: It is assumed the first half of rows are ``h′`` and the second half are ``β``.
- `data`: Stacked vector of observations `[amps...; phases...]`.
- `num_cells`: Number of grid cells. Usually this is a divisor of `nstates`.
- `R`: Vector of ``σ²`` for data. This is the diagonal of ``R``.

# References

[^1]: B. R. Hunt, E. J. Kostelich, and I. Szunyogh, “Efficient data assimilation for
    spatiotemporal chaos: A local ensemble transform Kalman filter,” Physica D: Nonlinear
    Phenomena, vol. 230, no. 1, pp. 112–126, Jun. 2007.
"""
function LETKF_measupdate(f, xb, data, R, y_grid, x_grid, pathnames, ens_size;
        ρ=1.1, localization=nothing, datatypes=(:amp, :phase))
    num_cells = length(y_grid)*length(x_grid)
    gridshape = (length(y_grid), length(x_grid))
    if !isnothing(localization)
        size(localization) == (num_cells, length(pathnames)) ||
            throw(ArgumentError("Size of `localization` must be `(num_cells, num_paths)`"))
    end

    # 1.
    yb = f(xb)  # f should handle phase wrap

    # Make sure xb, yb, and data are KeyedArrays for easier book keeping
    xb = KeyedArray(xb; field=SVector(:h, :b), y=y_grid, x=x_grid, ens=1:ens_size)
    yb = KeyedArray(yb; field=SVector(:amp, :phase), path=pathnames, ens=1:ens_size)
    data = KeyedArray(data; field=SVector(:amp, :phase), path=pathnames)

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
    for n in 1:num_cells
        yidx, xidx = CI[n][1], CI[n][2]

        # Currently localization is binary (cell is included or not)
        if isnothing(localization)
            loc_mask = trues(length(pathnames))
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

        data_loc = data(path=Index(loc_mask))

        if :amp in datatypes && :phase in datatypes
            Y_loc = [Y_loc(:amp); Y_loc(:phase)]
            Rinv_loc = Diagonal([fill(1/R[1], count(loc_mask)); fill(1/R[2], count(loc_mask))])
        elseif :amp in datatypes
            Rinv_loc = Diagonal(fill(1/R[1], count(loc_mask)))
        elseif :phase in datatypes
            Rinv_loc = Diagonal(fill(1/R[2], count(loc_mask)))
        end

        # 4.
        # Localization regularization here (HXf' = Yb)
        # from NergerLars2012
        # NOTE: Assumes diagonal R!
        dreg = 1

        C = AxisKeys.keyless(Y_loc)'*Rinv_loc*dreg

        # 5.
        # Can apply ρ here if H is linear, or if ρ is close to 1
        Patilde = inv((ens_size - 1)*I/ρ + C*Y_loc)

        # 6.
        # Symmetric square root
        Wa = sqrt((ens_size - 1)*Hermitian(AxisKeys.unname(Patilde)))

        # 7.
        if :amp in datatypes && :phase in datatypes
            Δ = [data_loc(:amp) .-  ybar_loc(:amp); phasediff.(data_loc(:phase), ybar_loc(:phase))]
        elseif :amp in datatypes
            Δ = data_loc(:amp) .- ybar_loc(:amp)
        elseif :phase in datatypes
            Δ = phasediff.(data_loc(:phase), ybar_loc(:phase))
        end

        wabar = Patilde*C*Δ
        wa = Wa .+ wabar

        # 8.
        xbbar_loc = xbbar(y=Index(yidx), x=Index(xidx))
        Xb_loc = Xb(y=Index(yidx), x=Index(xidx))

        xa(y=Index(yidx), x=Index(xidx)) .= Xb_loc*wa .+ xbbar_loc
    end

    return xa, yb
end

"""
    LETKF_measupdate(f, xb::KeyedArray, data::KeyedArray, R;
        ρ=1.1, localization=nothing, datatypes=(:amp, :phase))
"""
function LETKF_measupdate(f, xb::KeyedArray, data::KeyedArray, R;
        ρ=1.1, localization=nothing, datatypes=(:amp, :phase))
    y_grid, x_grid, = xb.y, xb.x
    pathnames = data.path
    ens_size = length(xb.ens)
    LETKF_measupdate(f, xb, data, R, y_grid, x_grid, pathnames, ens_size; ρ, localization, datatypes)
end

end # module
