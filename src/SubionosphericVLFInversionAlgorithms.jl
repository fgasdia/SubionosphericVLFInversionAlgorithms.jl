module SubionosphericVLFInversionAlgorithms

using Random
using ProgressMeter

export vfsa

const RNG = MersenneTwister(1234)


"""
    vfsa(f, x, xmin, xmax, Ta, Tm, NK, NT; saveprogress=:false, rng=RNG)

Apply very fast simulated annealing (VFSA) to the function `f`, returning a tuple of the
best `x` and corresponding energy `E = f(x)`. 

# Arguments

- `f`: objective function (to minimize) of `x`. Must return a scalar value.
- `x`: initial parameter (model) values.
- `xmin`: lower bound of each model parameter.
- `xmax`: upper bound of each model parameter.
- `Ta`: temperature function of iteration `k` for acceptance criterion.
- `Tm`: temperature function of iteration `k`, indexable for each model parameter.
- `NK`: number of iterations (changes in temperature).
- `NT`: number of moves at each temperature.
- `saveprogress=:false`:
    If `saveprogress` is `:false`, return `(x, E)`. If `:all`, return
    `(x, E, xprogress, Eprogress)` where the "progress" variables save every intermediate
    value of `x` and `E`.
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
function vfsa(f, x, xmin, xmax, Ta, Tm, NK, NT; saveprogress=:false, rng=RNG)
    length(x) == length(xmin) == length(xmax) ||
        throw(ArgumentError("`x`, `xmin`, and `xmax` must have same length"))
    all(xmin .< xmax) || throw(ArgumentError("`xmin` must be less than `xmax`"))

    x = copy(x)  # so `x` isn't modified in place
    NM = length(x)
    x′ = similar(x)

    E = f(x)

    if saveprogress == :false
        xprogress = nothing
        Eprogress = nothing
    elseif saveprogress == :all
        xprogress = [similar(x) for i = 1:NK*NT]
        Eprogress = Vector{typeof(E)}(undef, NK*NT)
        iter = 1
    end
    
    @showprogress for k in 1:NK
        Ta_k = Ta(k)
        Tm_k(i) = (t = Tm(k); t isa Number ? t : t[i])

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
                xprogress[iter] .= x
                Eprogress[iter] = E
                iter += 1
            end 
        end
    end

    return x, E, xprogress, Eprogress
end

end # module
