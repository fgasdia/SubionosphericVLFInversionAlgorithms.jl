module SubionosphericVLFInversionAlgorithms

using Random, Statistics, LinearAlgebra
using StaticArrays, AxisKeys, Distributions
using GeographicLib, LibGEOS, Proj4
using ScatteredInterpolation, GeoStats
using ProgressMeter

export vfsa, LETKF_measupdate
export ensemble_model
export gaspari1999_410, lonlatgrid_dists, obs2grid_diamondpill
export pathname

const RNG = MersenneTwister(1234)

(@isdefined wgs84) || const wgs84 = Projection("+proj=longlat +datum=WGS84 +no_defs")

# ESRI:102010, North America Equidistant Conic
(@isdefined esri_102010) || const esri_102010 = Projection("+proj=eqdc +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")

include("utils.jl")
include("forwardmodel.jl")

include("simulatedannealing.jl")
include("kalmanfilter.jl")

end # module
