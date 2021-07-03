using Test, Random, DelimitedFiles, Dates, LinearAlgebra
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

function testscenario(dr=500e3)
    dt = DateTime(2019, 2, 15, 18, 30)
    tx = TRANSMITTER[:NAA]
    rx = Receiver("Boulder", 40.02, -105.27, 0.0, VerticalDipole())
    paths = [(tx, rx)]

    west, east = -109.5, -63
    south, north = 36.5, 48.3
    x_grid, y_grid = build_xygrid(west, east, south, north, wgs84(), esri_102010(); dr)

    x = KeyedArray(fill(NaN, 2, length(y_grid), length(x_grid)); field=[:h, :b], y=y_grid, x=x_grid)
    x(:h) .= 70.0
    x(:b) .= 0.4

    return x, paths, dt
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

    @testset "Localization and grids" begin
        @info "Testing localization and grids"
        test_grids()
    end

    @testset "Kalman filter" begin
        @info "Testing Kalman filter"
        test_letkf(dayscenario)
    end
end
