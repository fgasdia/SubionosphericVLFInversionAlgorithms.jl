"""
    pathname(p)

Return path name string for (transmitter, receiver) path tuple `p`.
"""
pathname(p) = p[1].name*"-"*p[2].name

"""
    phasediff(a, b; deg=false)

Compute the smallest angle `a - b` in radians if `deg=false`, otherwise degrees.
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

"""
    strip(m::KeyedArray)
    strip(m::NamedDimsArray)

Remove named dims and axis keys from `m`, returning a view of the underlying array.
"""
Base.strip(m::KeyedArray) = AxisKeys.keyless(AxisKeys.unname(m))
Base.strip(m::NamedDimsArray) = AxisKeys.unname(m)

"""
l2norm(r)

Compute the *squared* L2 norm, ``||r||₂² = r₁² + r₂² + … + rₙ²``.

This would normalize the sum of squared residuals, which is what oocurs in the least squares
problem.
"""
function l2norm(r)
    return sum(abs2, r)
end

"""
l1norm(r)

Compute the L1 norm ``||r||₁ = |r₁| + |r₂| + … + |r₃|``.
"""
function l1norm(r)
    return sum(abs, r)
end

"""
hubernorm(r, ϵ)

Compute the Huber norm, which is the L2 norm squared between `-ϵ` and `ϵ` and the L1 norm
outside these bounds.

Guitton and Symes 2003 Robust inversion...
"""
function huber(r, ϵ)
    M(x, ϵ) = abs(x) <= ϵ ? x^2/(2*ϵ) : abs(x) - ϵ/2

    return sum(x->M(x, ϵ), r)
end

"""
tikhonov_gradient(m, mp, λh, λb)

Compute Tikhonov regularization of the gradient of model `m` with ``h′`` scaled by `λh` and
``β`` by `λb`.
"""
function tikhonov_gradient(m, mp, λh, λb)
    mp = ModelParams(mp.pathstep, 100e3, mp.r)
    _, _, h_grid, b_grid = dense_grid(m, mp)

    h_gy, h_gx = diff(h_grid; dims=1), diff(h_grid; dims=2)
    b_gy, b_gx = diff(b_grid; dims=1), diff(b_grid; dims=2)

    h_gy, h_gx = filter(!isnan, h_gy), filter(!isnan, h_gx)
    b_gy, b_gx = filter(!isnan, b_gy), filter(!isnan, b_gx)

    # Because diff behaves poorly at edges, let's use robust stats and remove a few outliers
    # (not a significant problem with large Δ and we don't want to miss actually big diffs)
    # for f in (h_gy, h_gx, b_gy, b_gx)
    #     mi, ma = quantile(f, [0.05, 0.95])
    #     filter!(x->mi < x < ma, f)
    # end

    return λh*(norm(h_gx, 2) + norm(h_gy, 2)) + λb*(norm(b_gx, 2) + norm(b_gy, 2))
end

"""
totalvariation(m, mp, μh, μb, αh, αb)

Compute total variation regularization of model `m` with regularization parameter `μ` and a
small stabilization term `α` such that
```
J(m) = μ ||∇m||₁ = μ Σ √( mx² + my² + α² )
```
"""
function totalvariation(itp, m, μh, μb, αh, αb)
    # mp = ModelParams(mp.pathstep, 100e3, mp.r, mp.w)  # `mp.w` is constrained by Δ of initial `mp`
    _, _, h_grid, b_grid = dense_grid(m, mp)




    h_gy, h_gx = diff(h_grid; dims=1), diff(h_grid; dims=2)
    b_gy, b_gx = diff(b_grid; dims=1), diff(b_grid; dims=2)

    h_gy, h_gx = filter(!isnan, h_gy), filter(!isnan, h_gx)
    b_gy, b_gx = filter(!isnan, b_gy), filter(!isnan, b_gx)

    αh² = αh^2
    αb² = αb^2

    h_total = 0.0
    b_total = 0.0
    for i in eachindex(h_gy)
        h_total += sqrt(h_gy[i]^2 + h_gx[i]^2 + αh²)
        b_total += sqrt(b_gy[i]^2 + b_gx[i]^2 + αb²)
    end

    return μh*h_total + μb*b_total
end
