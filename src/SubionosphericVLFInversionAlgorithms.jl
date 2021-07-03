module SubionosphericVLFInversionAlgorithms

using Random, Statistics, LinearAlgebra, Dates
using StaticArrays, AxisKeys, Distributions
using GeographicLib, LibGEOS, Proj4
using ScatteredInterpolation, GeoStats, Interpolations
using LongwaveModePropagator
const LMP = LongwaveModePropagator
using ProgressMeter

using LMPTools, PropagationModelPrep

export vfsa, LETKF_measupdate
export model, ensemble_model!, ScatteredInterpolant, GeoStatsInterpolant
export wgs84, esri_102010
export gaspari1999_410, lonlatgrid_dists, obs2grid_diamondpill, obs2grid_distance, anylocal
export modgaussian, build_xygrid, pathname, densify, gaussianstddev, compactlengthscale
export dense_grid, mediandr
export totalvariation, tikhonov_gradient

const RNG = MersenneTwister(1234)

project_path(parts...) = normpath(@__DIR__, "..", parts...)

wgs84() = Projection("+proj=longlat +datum=WGS84 +no_defs")

# ESRI:102010, North America Equidistant Conic
esri_102010() = Projection("+proj=eqdc +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")

include("utils.jl")
include("forwardmodel.jl")
include("localization_grids.jl")

include("simulatedannealing.jl")
include("kalmanfilter.jl")

end # module
