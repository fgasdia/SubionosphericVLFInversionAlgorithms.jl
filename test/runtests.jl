using Test, Random, DelimitedFiles, Dates
using AxisKeys, Distributions, Proj4
using ScatteredInterpolation, GeoStats
using LongwaveModePropagator

using LMPTools
using SubionosphericVLFInversionAlgorithms
const SIA = SubionosphericVLFInversionAlgorithms

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f_univariate(x) = 2only(x)^2 + 3only(x) + 1
parabola(x) = only(x)^2

include("utils.jl")
include("simulatedannealing.jl")
include("kalmanfilter.jl")

function test_models()
    x, paths, dt = testscenario()
    tx, rx = only(paths)
    hprimes, betas = strip(x(:h)), strip(x(:b))
    
    xygrid = SIA.build_xygrid(x)

    # ## ScatteredInterpolant
    itp = ScatteredInterpolant(ThinPlate(), esri_102010())

    hitp = ScatteredInterpolation.interpolate(itp.method, xygrid, vec(hprimes))
    bitp = ScatteredInterpolation.interpolate(itp.method, xygrid, vec(betas))
    input = SIA.model_observation(itp, hitp, bitp, tx, rx, dt)

    # Make sure fields are filled in appropriately
    @test all(x->isapprox(x, 70, atol=0.1), input.hprimes)
    @test all(x->isapprox(x, 0.4, atol=0.01), input.betas)
    @test input.frequency == 24e3
    @test range(tx, rx) < input.output_ranges[end] < 2*range(tx, rx)
    @test all(0 .< input.ground_epsrs .<= 81)

    a, p = model(itp, x, paths, dt; lwpc=false, pathstep=500e3)
    @test length(a) == length(p) == 1
    @test abs(only(p)) <= 2π

    if Sys.iswindows() && isfile("C:\\LWPCv21\\lwpm.exe")
        al, pl = model(itp, x, paths, dt; lwpc=true, pathstep=500e3)
        @test length(al) == length(pl) == 1
        @test abs(only(pl)) <= 2π
        @test rad2deg(abs(only(p - pl))) < 1
        @test abs(only(a - al)) < 1
    else
        @info " Skipping LWPC"
    end

    # ## GeoStatsInterpolant
    coords = reinterpret(reshape, Tuple{Float64,Float64}, xygrid)  # requires Julia v1.6
    τ = 2e-7*500e3
    f(h) = exp(-h^2/(2*τ^2))
    solver = LWR(
        :h′ => (weightfun=f,),
        :β => (weightfun=f,)
    )
    itp = GeoStatsInterpolant(solver, esri_102010(), coords)

    geox = georef((h′=hprimes, β=betas), PointSet(itp.coords))
    input = SIA.model_observation(itp, geox, tx, rx, dt)

    # Make sure fields are filled in appropriately
    @test all(input.hprimes .≈ 70)
    @test all(input.betas .≈ 0.4)
    @test input.frequency == 24e3
    @test range(tx, rx) < input.output_ranges[end] < 2*range(tx, rx)
    @test all(0 .< input.ground_epsrs .<= 81)

    a, p = model(itp, x, paths, dt; lwpc=false, pathstep=500e3)
    @test length(a) == length(p) == 1
    @test abs(only(p)) <= 2π

    if Sys.iswindows() && isfile("C:\\LWPCv21\\lwpm.exe")
        al, pl = model(itp, x, paths, dt; lwpc=true, pathstep=500e3)
        @test length(al) == length(pl) == 1
        @test abs(only(pl)) <= 2π
        @test rad2deg(abs(only(p - pl))) < 1
        @test abs(only(a - al)) < 1
    else
        @info " Skipping LWPC"
    end

    # ## hbfcn
    hbfcn(lo, la, dt) = ferguson(la, zenithangle(la, lo, dt), dt)
    input = SIA.model_observation(hbfcn, tx, rx, dt)

    @test all(x->isapprox(x, 73, atol=1), input.hprimes)
    @test all(x->isapprox(x, 0.3, atol=0.1), input.betas)
    @test input.frequency == 24e3
    @test range(tx, rx) < input.output_ranges[end] < 2*range(tx, rx)
    @test all(0 .< input.ground_epsrs .<= 81)

    a, p = model(hbfcn, paths, dt; lwpc=false, pathstep=500e3)
    @test length(a) == length(p) == 1
    @test abs(only(p)) <= 2π

    if Sys.iswindows() && isfile("C:\\LWPCv21\\lwpm.exe")
        al, pl = model(hbfcn, paths, dt; lwpc=true, pathstep=500e3)
        @test length(al) == length(pl) == 1
        @test abs(only(pl)) <= 2π
        @test rad2deg(abs(only(p - pl))) < 1
        @test abs(only(a - al)) < 1
    else
        @info " Skipping LWPC"
    end
end


@testset "SubionosphericVLFInversionAlgorithms" begin
    @info "Testing SubionosphericVLFInversionAlgorithms"

    @testset "Simulated annealing" begin
        @info "Testing simulated annealing"
        test_rosenbrock()
        test_univariate()
        test_parabola()
    end

    @testset "Forward models" begin
        @info "Testing forward models"
        @info " This may take a minute..."
        test_models()
    end

    @testset "Kalman filter" begin
        @info "Testing Kalman filter"
        test_arguments()
        test_day()
    end
end
