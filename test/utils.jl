function test_totalvariation()
    x, paths, dt = testscenario()
    tx, rx = only(paths)
    hprimes, betas = strip(x(:h)), strip(x(:b))
    
    xygrid = SIA.build_xygrid(x)

    coords = reinterpret(reshape, Tuple{Float64,Float64}, xygrid)  # requires Julia v1.6
    τ = 2e-7*500e3
    f(h) = exp(-h^2/(2*τ^2))
    solver = LWR(
        :h′ => (weightfun=f,),
        :β => (weightfun=f,)
    )
    itp = GeoStatsInterpolant(solver, esri_102010(), coords)

    totalvariation(m, mp, μh, μb, αh, αb)
end
