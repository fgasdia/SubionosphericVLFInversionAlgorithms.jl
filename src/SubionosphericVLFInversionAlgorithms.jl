module SubionosphericVLFInversionAlgorithms

using Random, Statistics, LinearAlgebra, Dates
using StaticArrays, AxisKeys, Distributions
using GeographicLib, LibGEOS, Proj
using NLopt, ImageFiltering
using ImageFiltering: KernelFactors
using ScatteredInterpolation, GeoStats, Interpolations
using LongwaveModePropagator
const LMP = LongwaveModePropagator
using ProgressMeter

using LMPTools, PropagationModelPrep

export vfsa, LETKF_measupdate, nlopt_estimate
export model, ensemble_model!, ScatteredInterpolant, GeoStatsInterpolant, lonlatmodel
export wgs84, esri_102010
export gaspari1999_410, lonlatgrid_dists, obs2grid_diamondpill, obs2grid_distance, anylocal,
    modgaussian, build_xygrid, pathname, densify, gaussianstddev, compactlengthscale,
    dense_grid, mediandr, filterbounds!, obs2grid_distances
export totalvariation, tikhonov_gradient, l2norm, objective, hubernorm, pseudohubernorm
export phasediff

const RNG = MersenneTwister(1234)

project_path(parts...) = normpath(@__DIR__, "..", parts...)

wgs84() = "+proj=longlat +datum=WGS84 +no_defs"

# ESRI:102010, North America Equidistant Conic
# esri_102010() = "+proj=eqdc +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
esri_102010() = "ESRI:102010"

include("utils.jl")
include("forwardmodel.jl")
include("localization_grids.jl")

include("simulatedannealing.jl")
include("kalmanfilter.jl")
include("nlopt.jl")

end # module
