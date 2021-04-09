using Test, Random
using SubionosphericVLFInversionAlgorithms

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f_univariate(x) = 2x[1]^2+3x[1]+1
parabola(x) = x[1]^2

function test_rosenbrock()
    x0 = [0.3, 0.8]
    T(k) = 200*exp(-2*k^(1/2))
    
    xbest, Ebest = vfsa(rosenbrock, x0, [-5, -5], [5, 5], T, T, 400, 50)

    @test xbest ≈ [1, 1] atol=1e-2
    @test_throws ArgumentError vfsa(rosenbrock, x0, -100, 100, T, T, 100, 1)
    @test_throws ArgumentError vfsa(rosenbrock, x0, [100, 100], [-100, -100], T, T, 100, 1)

    # Test if `rng` argument works
    Random.seed!(SubionosphericVLFInversionAlgorithms.RNG, 1234)
    x1, E1 = vfsa(rosenbrock, x0, [-5, -5], [5, 5], T, T, 400, 50)
    x2, E2 = vfsa(rosenbrock, x0, [-5, -5], [5, 5], T, T, 400, 50; rng=MersenneTwister(1234))
    @test x1 == x2
    @test E1 == E2
end

function test_univariate()
    x0 = [0.0]
    T(k) = 10*exp(-k)

    xbest, Ebest = vfsa(f_univariate, x0, -2, 1, T, T, 50, 5)

    @test only(xbest) ≈ -0.75 atol=1e-3
end

function test_parabola()
    T(k) = 10*exp(-k)

    x0 = [0.8]
    xbest, Ebest = vfsa(parabola, x0, -1, 1, T, T, 100, 3)

    @test only(xbest) ≈ 0 atol=1e-3
end

@testset "SubionosphericVLFInversionAlgorithms" begin
    test_rosenbrock()
    test_univariate()
    test_parabola()
end
