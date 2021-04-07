module SubionosphericVLFInversionAlgorithms

using Random

const RNG = MersenneTwister(1234)


"""
NM model parameters

Can return a vector (this is indexed) if different c and T0
"""
function T(k)
    return T0*exp(-c*k^(1/NM))
end


"""
    f: objective function (to minimize)
    x: initial parameter values
    T: transition distribution (we will call rand(T))
    t: annealing schedule
    kmax: number of iterations
    N: number of moves at const temp

# References

Algorithms for Optimization, Algorithm 8.4
Global Optimization Methods in Geophysical Inversion, Fig 4.11
"""
function vfsa(f, x, T, kmax, N)
    NM = length(x)
    x′ = similar(x)

    xbest .= x
    Ebest = E

    E = f(x)

    for k in 1:kmax
        Tk = T(k)
        Tmk = Tm(k)

        for n in 1:N
            u = rand(RNG, nparams)
            for i in 1:NM
                xnew = typemax(x[i])  # keep trying until new estimate is in bounds
                while !(xmin[i] <= xnew <= xmax[i]) 
                    y = sign(u[i] - 1/2)*Tmk[i]*((1 + 1/Tmk[i])^abs(2*u[i] - 1) - 1)
                    xnew = x[i] + y*(xmax[i] - xmin[i])
                end
                x′[i] = xnew
            end

            E′ = f(x′)
            ΔE = E′ - E

            if ΔE <= 0 || rand(RNG) < exp(-ΔE/Tk)
                x .= x′
                E = E′
            end

            if E′ < Ebest
                xbest, Ebest = x′, E′
            end
        end
    end

    return xbest, Ebest
end

end # module
