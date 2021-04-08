module SubionosphericVLFInversionAlgorithms

using Random

export vfsa

const RNG = MersenneTwister(1234)


"""
NM model parameters

Can return a vector (this is indexed) if different c and T0
"""
function T(k)
    return T0*exp(-c*k^(1/NM))
end


"""
    vfsa(f, x, xmin, xmax, Ta, Tm, NK, NT)

Apply very fast simulated annealing (VFSA) to the function `f`, returning a tuple of the
best `x` and corresponding energy `f(x)`. 

# Arguments

- `f`: objective function of (to minimize) of `x`. Must return a scalar value.
- `x`: initial parameter (model) values.
- `xmin`: lower bound of each model parameter.
- `xmax`: upper bound of each model parameter.
- `Ta`: temperature function of iteration `k` for acceptance criterion.
- `Tm`: temperature function of iteration `k`, indexable for each model parameter.
- `NK`: number of iterations.
- `NT`: number of moves at each temperature.

# References

Algorithms for Optimization, Algorithm 8.4
Global Optimization Methods in Geophysical Inversion, Fig 4.11
"""
function vfsa(f, x, xmin, xmax, Ta, Tm, NK, NT)
    length(x) == length(xmin) == length(xmax) ||
        throw(ArgumentError("`x`, `xmin`, and `xmax` must have same length"))
    all(xmin .< xmax) || throw(ArgumentError("`xmin` must be less than `xmax`"))

    x = copy(x)
    NM = length(x)
    x′ = similar(x)

    E = f(x)

    xbest = copy(x)
    Ebest = E

    for k in 1:NK
        Ta_k = Ta(k)
        Tm_k(i) = (t = Tm(k); t isa Number ? t : t[i])

        for n in 1:NT
            for i in 1:NM
                xnew = typemax(x[i])
                while !(xmin[i] <= xnew <= xmax[i])
                    # keep trying until new estimate is in bounds (usually called only once)
                    u = rand(RNG)
                    y = sign(u - 1/2)*Tm_k(i)*((1 + 1/Tm_k(i))^abs(2*u - 1) - 1)
                    xnew = x[i] + y*(xmax[i] - xmin[i])
                end
                x′[i] = xnew
            end

            E′ = f(x′)
            ΔE = E′ - E

            if ΔE <= 0 || rand(RNG) < exp(-ΔE/Ta_k)
                x .= x′
                E = E′
            end

            if E′ < Ebest
                xbest .= x′
                Ebest = E′
            end
        end
    end

    return xbest, Ebest
end

end # module
