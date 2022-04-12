struct ScatteredInterpolant{T,T2,T3}
    method::T
    projection::T2
    coords::T3
end

struct GeoStatsInterpolant{T,T2,T3}
    method::T
    projection::T2
    coords::T3
end

"""
    pathpts(tx, rx; dist=100e3) → (line, wpts)

Return a `GeodesicLine` and waypoints `wpts` from `tx` to `rx` of path tuple `p`
with waypoints every `dist` meters along the path.
"""
function pathpts(tx, rx; dist=100e3)
    line = GeodesicLine(tx, rx)
    wpts = waypoints(line; dist=dist)[1:end-1]

    return line, wpts
end

"""
    _model_observation(tx, rx; pathstep=100e3) → (input, wpts)

Construct the baseline mutable `ExponentialInput` `input` and waypoints `wpts` with
transmitter `tx` and receiver `rx`.

The rest of the `input` should be filled in using [`model_observation`](@ref).
"""
function _model_observation(tx, rx; pathstep=100e3)
    _, wpts = pathpts(tx, rx; dist=pathstep)

    # Build the ExponentialInput
    input = ExponentialInput()
    input.name = "estimate"
    input.description = ""
    input.datetime = Dates.now()
    input.segment_ranges = Vector{Float64}(undef, length(wpts))
    input.hprimes = similar(input.segment_ranges)
    input.betas = similar(input.segment_ranges)
    input.b_mags = similar(input.segment_ranges)
    input.b_dips = similar(input.segment_ranges)
    input.b_azs = similar(input.segment_ranges)
    input.ground_sigmas = similar(input.segment_ranges)
    input.ground_epsrs = Vector{Int}(undef, length(wpts))
    input.frequency = tx.frequency.f

    # NOTE: if changing `output_ranges` step, must also update in `model`!
    input.output_ranges = collect(0:5e3:round(range(tx, rx)+10e3, digits=-4, RoundUp))
    
    return input, wpts
end

"""
    model_observation(itp, geox, tx, rx, datetime; pathstep=100e3)

Build the `ExponentialInput` for a single path from `tx` to `rx` at `datetime`.
"""
function model_observation(itp::GeoStatsInterpolant, geox, tx, rx, datetime; pathstep=100e3)
    :h in itp.method.varnames && :b in itp.method.varnames ||
        @warn "`:h` and `:b` should be defined in `itp.method`"

    input, wpts = _model_observation(tx, rx; pathstep)

    # Projected wpts
    pts = PointSet(permutedims(
        transform(wgs84(), itp.projection, [getindex.(wpts, :lon) getindex.(wpts, :lat)])
    ))

    problem = EstimationProblem(geox, pts, (:h, :b))
    solution = solve(problem, itp.method)

    geoaz = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).azi

    for i in 1:length(wpts)
        lat, lon, dist = wpts[i].lat, wpts[i].lon, wpts[i].dist
        
        bfield = igrf(geoaz, lat, lon, year(datetime))
        ground = GROUND[LMPTools.get_groundcode(lat, lon)]

        input.segment_ranges[i] = dist
        input.hprimes[i] = solution[:h][i]
        input.betas[i] = solution[:b][i]
        input.b_mags[i] = bfield.B
        input.b_dips[i] = LMP.dip(bfield)
        input.b_azs[i] = LMP.azimuth(bfield)
        input.ground_sigmas[i] = ground.σ
        input.ground_epsrs[i] = ground.ϵᵣ
    end

    return input
end

"""
    model_observation(itp, hitp, bitp, tx, rx, datetime; pathstep=100e3)

Build the `ExponentialInput` for a single path from `tx` to `rx` at `datetime`.
"""
function model_observation(itp::ScatteredInterpolant, hitp, bitp, tx, rx, datetime; pathstep=100e3)
    input, wpts = _model_observation(tx, rx; pathstep)

    # Projected wpts
    pts = permutedims(
        transform(wgs84(), itp.projection, [getindex.(wpts, :lon) getindex.(wpts, :lat)])
    )

    geoaz = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).azi

    for i in 1:length(wpts)
        lat, lon, dist = wpts[i].lat, wpts[i].lon, wpts[i].dist
        
        bfield = igrf(geoaz, lat, lon, year(datetime))
        ground = GROUND[LMPTools.get_groundcode(lat, lon)]

        input.segment_ranges[i] = dist
        input.hprimes[i] = only(ScatteredInterpolation.evaluate(hitp, pts[:,i]))
        input.betas[i] = only(ScatteredInterpolation.evaluate(bitp, pts[:,i]))
        input.b_mags[i] = bfield.B
        input.b_dips[i] = LMP.dip(bfield)
        input.b_azs[i] = LMP.azimuth(bfield)
        input.ground_sigmas[i] = ground.σ
        input.ground_epsrs[i] = ground.ϵᵣ
    end

    return input
end

"""
    model_observation(hbfcn, tx, rx, datetime; pathstep=100e3)

Build the `ExponentialInput` using `hbfcn`, a function of `(lon, lat, datetime)`, to compute
`(h′, β)` between the transmitter `tx` and receiver `rx`.

This uses the [`igrf`](@ref) magnetic field and ground code from LMPTools.jl.
"""
function model_observation(hbfcn, tx, rx, datetime; pathstep=100e3)
    input, wpts = _model_observation(tx, rx; pathstep)

    geoaz = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).azi

    for i in 1:length(wpts)
        lat, lon, dist = wpts[i].lat, wpts[i].lon, wpts[i].dist
        
        bfield = igrf(geoaz, lat, lon, year(datetime))
        ground = GROUND[LMPTools.get_groundcode(lat, lon)]

        h, b = hbfcn(lon, lat, datetime)

        input.segment_ranges[i] = dist
        input.hprimes[i] = h
        input.betas[i] = b
        input.b_mags[i] = bfield.B
        input.b_dips[i] = LMP.dip(bfield)
        input.b_azs[i] = LMP.azimuth(bfield)
        input.ground_sigmas[i] = ground.σ
        input.ground_epsrs[i] = ground.ϵᵣ
    end

    return input
end

"""
    model(itp, x, paths, datetime; pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1)

Return the model observations `(amps, phases)` for spatial interpolation method `itp` and
`datetime`.

If `x` is a `KeyedArray`, then it is transformed to a vector where the first half is ``h′``
and the second half is ``β``.

Uses LWPC as the forward model if `lwpc` is true; otherwise, uses LongwaveModePropagator.jl.
`numexe` specifies the number of LWPC executables to use.
"""
function model(itp::GeoStatsInterpolant, x, paths, datetime;
    pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1)

    npts = length(x) ÷ 2
    hprimes = x[1:npts]
    betas = x[npts+1:end]

    geox = georef((h=filter(!isnan, hprimes), b=filter(!isnan, betas)), PointSet(itp.coords))

    batch = BatchInput{ExponentialInput}()
    batch.name = "estimate"
    batch.description = ""
    batch.datetime = Dates.now()
    batch.inputs = Vector{ExponentialInput}(undef, length(paths))

    for i in eachindex(paths)
        tx, rx = paths[i]
        input = model_observation(itp, geox, tx, rx, datetime; pathstep)
        batch.inputs[i] = input
    end

    if lwpc
        computejob = LocalParallel("estimate", ".", "C:\\LWPCv21\\lwpm.exe", numexe, 90)
        output = LWPC.run(batch, computejob; savefile=false, sleeptime=sleeptime)
    else
        output = LMP.buildrun(batch; params=LMPParams(approxsusceptibility=true))
    end

    amps = Vector{Float64}(undef, length(paths))
    phases = similar(amps)

    for i in eachindex(paths)
        tx, rx = paths[i]
        d = range(tx, rx)
        o = output.outputs[i]

        # NOTE: step size here should match `output_ranges` step in `model_observation`!
        aitp = LinearInterpolation(0:5e3:last(o.output_ranges), o.amplitude)
        pitp = LinearInterpolation(0:5e3:last(o.output_ranges), o.phase)
        amps[i] = aitp(d)
        phases[i] = pitp(d)
    end

    return amps, phases
end
model(itp::GeoStatsInterpolant, x::KeyedArray, paths, datetime; pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1) =
    model(itp, [x(:h); x(:b)], paths, datetime; pathstep, lwpc, numexe, sleeptime)

function model(itp::ScatteredInterpolant, x, paths, datetime;
    pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1)

    npts = length(x) ÷ 2
    hprimes = x[1:npts]
    betas = x[npts+1:end]
    
    hitp = ScatteredInterpolation.interpolate(itp.method, itp.coords, filter(!isnan, hprimes))
    bitp = ScatteredInterpolation.interpolate(itp.method, itp.coords, filter(!isnan, betas))

    batch = BatchInput{ExponentialInput}()
    batch.name = "estimate"
    batch.description = ""
    batch.datetime = Dates.now()
    batch.inputs = Vector{ExponentialInput}(undef, length(paths))

    for i in eachindex(paths)
        tx, rx = paths[i]
        input = model_observation(itp, hitp, bitp, tx, rx, datetime; pathstep)
        batch.inputs[i] = input
    end

    if lwpc
        computejob = LocalParallel("estimate", ".", "C:\\LWPCv21\\lwpm.exe", numexe, 90)
        output = LWPC.run(batch, computejob; savefile=false, sleeptime=sleeptime)
    else
        output = LMP.buildrun(batch; params=LMPParams(approxsusceptibility=true))
    end

    amps = Vector{Float64}(undef, length(paths))
    phases = similar(amps)
    for i in eachindex(paths)
        tx, rx = paths[i]
        d = range(tx, rx)
        o = output.outputs[i]

        # NOTE: step size here should match `output_ranges` step in `model_observation`!
        aitp = LinearInterpolation(0:5e3:last(o.output_ranges), o.amplitude)
        pitp = LinearInterpolation(0:5e3:last(o.output_ranges), o.phase)
        amps[i] = aitp(d)
        phases[i] = pitp(d)
    end

    return amps, phases
end
model(itp::ScatteredInterpolant, x::KeyedArray, paths, datetime; pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1) =
    model(itp, [vec(x(:h)); vec(x(:b))], paths, datetime; pathstep, lwpc, numexe, sleeptime)

"""
    model(hbfcn::Function, paths, datetime; pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1)

Use `hbfcn`, a function of `(lon, lat, datetime)`, to compute `(h′, β)` along each 
vector of (transmitter, receiver) `paths`.

`pathstep` is the segment length in meters along each path.

If `lwpc` is `true`, then LWPC is used as the forward model. If `lwpc` is false, then
LongwaveModePropagator is used as the forward model. As of `v0.2.0` of LongwaveModePropagator,
it is significantly slower than LWPC when using many segemnts and thus not preferred for
ionosphere estimation. `numexe` specifies the number of LWPC executables to use.

See also: [`model_observation`](@ref)
"""
function model(hbfcn::Function, paths, datetime;
    pathstep=100e3, lwpc=true, numexe=16, sleeptime=0.1)

    batch = BatchInput{ExponentialInput}()
    batch.name = "estimate"
    batch.description = ""
    batch.datetime = Dates.now()
    batch.inputs = Vector{ExponentialInput}(undef, length(paths))

    for i in eachindex(paths)
        tx, rx = paths[i]
        input = model_observation(hbfcn, tx, rx, datetime; pathstep)
        batch.inputs[i] = input
    end

    if lwpc
        computejob = LocalParallel("estimate", ".", "C:\\LWPCv21\\lwpm.exe", numexe, 90)
        output = LWPC.run(batch, computejob; savefile=false, sleeptime=sleeptime)
    else
        output = LMP.buildrun(batch; params=LMPParams(approxsusceptibility=true))
    end

    amps = Vector{Float64}(undef, length(paths))
    phases = similar(amps)
    for i in eachindex(paths)
        tx, rx = paths[i]
        d = range(tx, rx)
        o = output.outputs[i]

        # Using LMP we could skip the LinearInterpolation and compute the field at exactly
        # the correct distance, but for consistency with LWPC we'll interpolate the output.
        # NOTE: step size here should match `output_ranges` step in `model_observation`!
        aitp = LinearInterpolation(0:5e3:last(o.output_ranges), o.amplitude)
        pitp = LinearInterpolation(0:5e3:last(o.output_ranges), o.phase)
        amps[i] = aitp(d)
        phases[i] = pitp(d)
    end

    return amps, phases
end

function lonlatsegment(lon, lat, dist, dt, hbfcn, nufcn, bfcn, gfcn)
    h, b = hbfcn(lon, lat, dt)
    return HomogeneousWaveguide(
        bfcn(lon, lat, dt),
        Species(LMP.QE, LMP.ME, z->waitprofile(z, h, b), z->nufcn(z, lon, lat)),
        gfcn(lon, lat),
        dist
    )
end

"""
    lonlatmodel

Run LMP over `paths` using functions of ``h′, β``, ``BField``, ``Ground``, and collision
frequency.

# Arguments

- `hbfcn(lon, lat, datetime) → h′, β`
- `nufcn(z, lon, lat) → ν`
- `bfcn(geoaz, lon, lat, datetime) → BField`
- `gfcn(lon, lat) → Ground`
"""
function lonlatmodel(hbfcn, nufcn, bfcn, gfcn, paths, datetime; pathstep=100e3)
    amps = Vector{Float64}(undef, length(paths))
    phases = similar(amps)
    for j in eachindex(paths)
        tx, rx = paths[j]
        _, wpts = pathpts(tx, rx; dist=pathstep)

        geoaz = inverse(tx.longitude, tx.latitude, rx.longitude, rx.latitude).azi

        bffcn(lon, lat, dt) = bfcn(geoaz, lon, lat, dt)
        wvg = SegmentedWaveguide([lonlatsegment(wpts[i].lon, wpts[i].lat, wpts[i].dist,
            datetime, hbfcn, nufcn, bffcn, gfcn) for i in eachindex(wpts)])

        gs = GroundSampler(range(tx, rx), Fields.Ez)
        _, a, p = propagate(wvg, tx, gs)
        amps[j] = a
        phases[j] = p
    end
    return amps, phases
end
