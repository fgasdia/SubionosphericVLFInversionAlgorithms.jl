using Test, Random, DelimitedFiles
using SubionosphericVLFInversionAlgorithms

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f_univariate(x) = 2only(x)^2 + 3only(x) + 1
parabola(x) = only(x)^2

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

    # saveprogress with multiple elements x
    xbest, Ebest, xprogress, Eprogress = vfsa(rosenbrock, x0, [-5, -5], [5, 5], T, T, 400, 50;
        saveprogress=:all)
    @test size(xprogress,1) == 400*50
    @test length(Eprogress) == 400*50
    @test xbest == xprogress[end,:]
    @test Ebest == last(Eprogress)

    # # filename
    @info "  Writing progress to file. This may take a while..."
    fname, _ = mktemp()
    xbest, Ebest, xprogress, Eprogress = vfsa(rosenbrock, x0, [-5, -5], [5, 5], T, T, 400, 50;
        saveprogress=:all, filename=fname)
    dat = readdlm(fname, ',')
    @test dat[:,4:end] == xprogress
    @test dat[:,3] == Eprogress
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

    # saveprogress with single element x
    xbest, Ebest, xprogress, Eprogress = vfsa(parabola, x0, -1, 1, T, T, 100, 3;
        saveprogress=:all)
    @test length(xprogress) == 100*3
    @test length(Eprogress) == 100*3
    @test only(xbest) == last(xprogress)
    @test Ebest == last(Eprogress)

    # filename
    fname, _ = mktemp()
    xbest, Ebest, xprogress, Eprogress = vfsa(parabola, x0, -1, 1, T, T, 100, 3;
        saveprogress=:all, filename=fname)
    dat = readdlm(fname, ',')
    @test dat[:,end] == xprogress[:]
    @test dat[:,3] == Eprogress
end

@testset "SubionosphericVLFInversionAlgorithms" begin
    @info "Testing SubionosphericVLFInversionAlgorithms"
    test_rosenbrock()
    test_univariate()
    test_parabola()
end
