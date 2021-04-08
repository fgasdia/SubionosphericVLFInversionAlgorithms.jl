using Test
using SubionosphericVLFInversionAlgorithms

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f_univariate(x) = 2x[1]^2+3x[1]+1
parabola(x) = x[1]^2

function test_rosenbrock()
    x0 = rand(2)
    # result = optimize(rosenbrock, x0, BFGS())

    T(k) = 200*exp(-k^(1/2))
    xbest, Ebest = vfsa(rosenbrock, x0, [-5, -5], [5, 5], T, T, 500, 50)

    @test xbest ≈ [1, 1] atol=1e-2
    @test_throws ArgumentError vfsa(rosenbrock, x0, -100, 100, T, T, 100, 1)
    @test_throws ArgumentError vfsa(rosenbrock, x0, [100, 100], [-100, -100], T, T, 100, 1)
end

function test_univariate()
    T(k) = 10*exp(-k)

    x0 = [0.0]
    xbest, Ebest = vfsa(f_univariate, x0, -2, 1, T, T, 200, 10)

    @test only(xbest) ≈ -0.75 atol=1e-2
end

function test_parabola()
    T(k) = 10*exp(-k)

    x0 = [0.8]
    xbest, Ebest = vfsa(parabola, x0, -1, 1, T, T, 100, 1)

    @test only(xbest) ≈ 0 atol=1e-6
end

@testset "SubionosphericVLFInversionAlgorithms" begin
    test_rosenbrock()
    test_univariate()
    test_parabola()
end
