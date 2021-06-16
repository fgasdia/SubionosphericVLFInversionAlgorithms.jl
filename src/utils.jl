"""
    gaspari_cohn99_410(z, c)

Compactly supported 5th-order piecewise rational function that resembles a Gaussian evaluated
over distances `z` with scale length `c`.

# References

[^1]: Gaspari Cohn 1999, Construction of correlation functions in two and three dimensions.
    Eqn 4.10
"""
function gaspari_cohn99_410(z, c)
    C0 = zeros(size(z))

    for i in eachindex(C0)
        tz = z[i]
        if 0 <= abs(tz) <= c
            C0[i] = -(1/4)*(abs(tz)/c)^5 + (1/2)*(tz/c)^4 + (5/8)*(abs(tz)/c)^3 -
                (5/3)*(tz/c)^2 + 1
        elseif c <= abs(tz) <= 2c
            C0[i] = (1/12)*(abs(tz)/c)^5 - (1/2)*(tz/c)^4 + (5/8)*(abs(tz)/c)^3 +
                (5/3)*(tz/c)^2 - 5*(abs(tz)/c) + 4 - (2/3)*c/abs(tz)
        # elseif 2c <= abs(tz)
        # C0[i] = 0
        end
    end

    return C0
end

"""
    phasediff(a, b; deg=false))

Compute the smallest angle `a - b` in radians.
"""
function phasediff(a, b; deg=false)
    if deg
        a, b = deg2rad(a), deg2rad(b)
    end

    d = mod2pi(a) - mod2pi(b)
    d = mod2pi(d + π) - π

    if deg
        d = rad2deg(d)
    end

    return d
end
