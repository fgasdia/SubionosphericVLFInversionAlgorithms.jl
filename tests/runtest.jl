using Test
using SubionosphericVLFInversionAlgorithms
using Optim

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function test_rosenbrock()
    x0 = zeros(2)
    # result = optimize(rosenbrock, x0, BFGS())

    T(k) = 50*exp(-k^(1/2))
    xbest, Ebest = vfsa(rosenbrock, x0, [-100, -100], [100, 100], T, T, 100, 1)

    @test xbest ≈ [1, 1]
    @test_throws ArgumentError vfsa(rosenbrock, x0, -100, 100, T, T, 100, 1)
    @test_throws ArgumentError vfsa(rosenbrock, x0, [100, 100], [-100, -100], T, T, 100, 1)
end

@testset "SubionosphericVLFInversionAlgorithms" begin
    test_rosenbrock()
end