function test_totalvariation()
    x, paths, _ = testscenario()
    
    xygrid = build_xygrid(x(:h))

    lengthscale = 500e3
    τ = 2e-7*lengthscale
    f(h) = exp(-h^2/(2*τ^2))
    solver = LWR(
        :h′ => (weightfun=f,),
        :β => (weightfun=f,)
    )
    itp = GeoStatsInterpolant(solver, esri_102010(), xygrid)

    localizationfcn(lola) = anylocal(obs2grid_distance(lola, paths; r=lengthscale))

    μh, μb = 1, 1
    αh, αb = 0, 0
    ϕ = totalvariation(itp, x, μh, μb, αh, αb; localizationfcn)
    @test ϕ ≈ 0 atol=1e-8
    ϕ = totalvariation(itp, x, μh, μb, αh, αb; localizationfcn=nothing)
    @test ϕ ≈ 0 atol=1e-8

    xc = copy(x)
    xc(:h) .= rand(size(xc(:h))...)
    xc(:b) .= rand(size(xc(:b))...)
    ϕ = totalvariation(itp, xc, μh, μb, αh, αb; localizationfcn)
    @test ϕ > 0
    ϕv = totalvariation(itp, [vec(xc(:h)); vec(xc(:b))], μh, μb, αh, αb; localizationfcn)
    @test ϕv ≈ ϕ
    ϕ2 = totalvariation(itp, xc, μh, μb, αh, αb; localizationfcn=nothing)
    @test ϕ2 > 0
    @test !isapprox(ϕ, ϕ2; atol=0.1)
end

function test_tikhonov()
    x, paths, _ = testscenario()
    
    xygrid = build_xygrid(x(:h))

    lengthscale = 500e3
    τ = 2e-7*lengthscale
    f(h) = exp(-h^2/(2*τ^2))
    solver = LWR(
        :h′ => (weightfun=f,),
        :β => (weightfun=f,)
    )
    itp = GeoStatsInterpolant(solver, esri_102010(), xygrid)

    localizationfcn(lola) = anylocal(obs2grid_distance(lola, paths; r=lengthscale))

    λh, λb = 1, 1
    ϕ = tikhonov_gradient(itp, x, λh, λb; localizationfcn)
    @test ϕ ≈ 0 atol=1e-8
    ϕ = tikhonov_gradient(itp, x, λh, λb; localizationfcn=nothing)
    @test ϕ ≈ 0 atol=1e-8

    xc = copy(x)
    xc(:h) .= rand(size(xc(:h))...)
    xc(:b) .= rand(size(xc(:b))...)
    ϕ = tikhonov_gradient(itp, xc, λh, λb; localizationfcn)
    @test ϕ > 0
    ϕ2 = tikhonov_gradient(itp, xc, λh, λb; localizationfcn=nothing)
    @test ϕ2 > 0
    @test !isapprox(ϕ, ϕ2; atol=0.1)
end
