# Lyot Coronagraph Simulation and Multi-Objective Optimization

## Optical model

The simulator implements a classical four-plane Lyot coronagraph in Python using POPPY's Fresnel/Fraunhofer propagation engine. Light propagates through:

- **(A)** a circular entrance pupil (6 m aperture),
- **(B)** a focal-plane occulting spot that blocks the on-axis PSF core,
- **(C)** a relayed pupil with an undersized Lyot stop that clips the starlight the occulter diffracts toward the pupil edge, and
- **(D)** the science detector.

Each plane-to-plane propagation is an FFT; free design parameters are the occulter radius and Lyot stop transmission fraction, both defined relative to the diffraction scale λ/D.

Two performance metrics are computed per design.

**Throughput** is aperture photometry (0.7 λ/D radius) on an off-axis source, normalized to the same measurement on a bare, coronagraph-free aperture, sampled across a grid of separations. The **inner working angle (IWA)** is the separation at which this throughput curve crosses 50%.

**Contrast** is the worst (maximum) azimuthally-averaged annulus of on-axis residual starlight between IWA and a fixed outer working angle (22 λ/D), normalized to the unocculted peak intensity — using the worst rather than the mean annulus prevents a design from hiding one bad region behind several good ones.

## Optimization: epsilon-constraint sweep

Contrast and throughput trade off directly, so no single design is optimal. Rather than scalarizing both into one weighted objective — which, in initial testing, was found to trace only the convex hull of the trade-off curve and collapse several distinct weightings onto the same design — the final approach uses an **epsilon-constraint formulation**: for each of several throughput floors ε, minimize contrast subject to throughput(x) ≥ ε, via `scipy.optimize.differential_evolution` with a nonlinear constraint.

A second constraint caps the IWA at 6 λ/D; without it, the optimizer exploits the fact that enlarging the occulter improves both nominal contrast and (because the throughput benchmark scales with IWA) nominal throughput simultaneously, at the cost of an unusably large inner working angle.

Because the contrast landscape is a bumpy, multi-modal function of occulter radius, the optimizer occasionally converges to a locally- rather than globally-optimal design at a given ε. A **non-dominated (Pareto) filter** is applied post hoc, removing any result that is matched or beaten on both metrics by another result in the same sweep — a standard, principled correction that requires no re-optimization.

## Two-resolution search/verify strategy

Each design evaluation requires ~19 PSF calculations, making the full sweep expensive at high angular sampling. The optimizer therefore searches at a coarse resolution and every candidate the sweep proposes as optimal is independently re-evaluated (forward pass only, no re-optimization) at a much finer resolution before being trusted.

This proved necessary rather than optional: contrast, in particular, showed real sensitivity to sampling resolution during development, while throughput was comparatively stable — motivating verification of every reported number rather than reporting search-phase values directly.

## Vortex coronagraph comparison

As a targeted comparison to the vector vortex/AGPM coronagraph technology developed at Uppsala, an idealized scalar vortex phase mask (topological charge 2, matching the AGPM lineage) was implemented as an alternative focal-plane optic within the same four-plane architecture, replacing the occulter with a phase-only element imparting `exp(i · 2 · θ)`.

One representative, verified classical-Lyot design was re-evaluated alongside the vortex mask at matched Lyot stop and at a higher sampling resolution than the classical baseline, motivated by an explicit finding: the vortex mask's null depth is strongly limited by sampling near its central phase singularity (measured suppression improved roughly 16–19× per doubling of oversampling in this work), unlike the classical occulter.

At matched Lyot stop, the vortex mask achieved an IWA of 1.9–2.3 λ/D against the classical design's ~5 λ/D, with roughly 15–130× better contrast at a moderate throughput cost — consistent with the sub-diffraction-limit inner working angles reported for AGPM vortex coronagraphs in the published VORTEX/NACO and Keck/NIRC2 literature associated with this research group.
