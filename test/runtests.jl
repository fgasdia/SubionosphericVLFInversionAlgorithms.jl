using Test, Random, DelimitedFiles, Dates
using AxisKeys, Distributions
using LongwaveModePropagator

using LMPTools
using SubionosphericVLFInversionAlgorithms

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f_univariate(x) = 2only(x)^2 + 3only(x) + 1
parabola(x) = only(x)^2

include("simulatedannealing.jl")

@testset "SubionosphericVLFInversionAlgorithms" begin
    @info "Testing SubionosphericVLFInversionAlgorithms"

    @testset "Simulated annealing" begin
        @info "Testing simulated annealing"
        test_rosenbrock()
        test_univariate()
        test_parabola()
    end
end
