using Test, Random, DelimitedFiles, Dates, LinearAlgebra
using AxisKeys, Distributions, Proj4
using ScatteredInterpolation, GeoStats
using LongwaveModePropagator
using UnPack

using LMPTools
using SubionosphericVLFInversionAlgorithms
const SIA = SubionosphericVLFInversionAlgorithms

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f_univariate(x) = 2only(x)^2 + 3only(x) + 1
parabola(x) = only(x)^2

include("utils.jl")
include("forwardmodel.jl")
include("localization_grids.jl")
include("simulatedannealing.jl")
include("kalmanfilter.jl")
include("nlopt.jl")

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

function buildpaths()
    transmitters = [TRANSMITTER[:NLK], TRANSMITTER[:NML]]
    receivers = [
        Receiver("Whitehorse", 60.724, -135.043, 0.0, VerticalDipole()),
        Receiver("Churchill", 58.74, -94.085, 0.0, VerticalDipole()),
        Receiver("Stony Rapids", 59.253, -105.834, 0.0, VerticalDipole()),
        Receiver("Fort Smith", 60.006, -111.92, 0.0, VerticalDipole()),
        Receiver("Bella Bella", 52.1675508, -128.1545219, 0.0, VerticalDipole()),
        Receiver("Nahanni Butte", 61.0304412, -123.3926734, 0.0, VerticalDipole()),
        Receiver("Juneau", 58.32, -134.41, 0.0, VerticalDipole()),
        Receiver("Ketchikan", 55.35, -131.673, 0.0, VerticalDipole()),
        Receiver("Winnipeg", 49.8822, -97.1308, 0.0, VerticalDipole()),
        Receiver("IslandLake", 53.8626, -94.6658, 0.0, VerticalDipole()),
        Receiver("Gillam", 56.3477, -94.7093, 0.0, VerticalDipole())
    ]
    paths = [(tx, rx) for tx in transmitters for rx in receivers]

    return paths
end

function dayscenario()
    ens_size = 30  # size of the ensemble... the number of ionospheres
    ntimes = 2  # how many time steps to take

    # The DateTime will be needed for the prior ionosphere and possibly the IGRF magnetic field
    dt = DateTime(2020, 3, 1, 20, 00)  # day

    modelproj = esri_102010()
    dr = 500e3  # modelproj grid spacing in meters, coarse for testing only
    pathstep = 500e3  # distance in WGS84 meters for path segments, coarse for testing
    lengthscale = 600e3  # ionosphere correlation length σ in WGS84

    # y_grid and x_grid (or `y` and `x`) are the coordinates of estimation grid points in
    # the y-axis and x-axis. These could be latitude and longitude, respectively, but it
    # is better to use a plane projection so that the grid points are more equally spaced.
    west, east = -135.5, -93
    south, north = 46, 63
    x_grid, y_grid = build_xygrid(west, east, south, north, wgs84(), modelproj; dr)

    truthfcn(lo, la, dt) = ferguson(la, zenithangle(la, lo, dt), dt)

    return (ens_size=ens_size, ntimes=ntimes, dt=dt, modelproj=modelproj, dr=dr,
        pathstep=pathstep, x_grid=x_grid, y_grid=y_grid, lengthscale=lengthscale,
        truthfcn=truthfcn)
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
        @info "    This may take a minute..."
        test_models()
    end

    @testset "Localization and grids" begin
        @info "Testing localization and grids"
        test_grids()
    end

    @testset "Utils" begin
        @info "Testing utils"
        test_totalvariation()
        test_tikhonov()
    end

    @testset "Kalman filter" begin
        @info "Testing Kalman filter"
        test_letkf(dayscenario)
    end

    @testset "NLopt" begin
        @info "Testing NLopt"
        test_nlopt(dayscenario)
    end
end
