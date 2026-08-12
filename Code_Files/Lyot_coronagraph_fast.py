# Lyot vs Vortex Coronagraph test

import time
import numpy as np
import matplotlib.pyplot as plt
import poppy
from astropy import units as u
from scipy.optimize import differential_evolution, NonlinearConstraint
 
poppy.conf.use_fftw = False
 

# ---- Base parameters ----
 
telescope_diameter = 6.0 * u.m
wavelength = 1.0 * u.micron
fov_arcsec = 2.0
FOV_PIXELS = 256
NATIVE_PIXELSCALE_ARCSEC = (2 * fov_arcsec) / FOV_PIXELS
 
APERTURE_RADIUS_LD = 0.7
OUTER_BOUND_LD = 16.0          
OWA_LD = 22.0
N_CONTRAST_BINS = 20
SEPARATIONS_LD = np.linspace(0.5, OUTER_BOUND_LD, 10)
THETA_DEG = 90.0
 
BOUNDS = [(1.5, 8.0), (0.5, 0.995)]
N_EPSILON_STEPS = 8
 
# My huge problem on this was the occulter pinning against the upper bound at EVERY epsilon target, even after widening. That's because nothing in the objective penalized a large
# I'm capping IWA directly: since I found this as a science requirement for imaging close-in planets
IWA_MAX_LD = 6.0
 
SEARCH_NPIX, SEARCH_OVERSAMPLE = 128, 2  # This regime for search sampling treated my old computer kindly 
VERIFY_NPIX, VERIFY_OVERSAMPLE = 256, 4

DE_POPSIZE = 12
DE_MAXITER = 25

# Matches the AGPM/vortex lineage (Mawet et al.): vortices carry even charges only 
VORTEX_CHARGE = 2               # charge 2 gives me the small IWA
VORTEX_NPIX, VORTEX_OVERSAMPLE = 256, 8  
 
 
class VortexPhaseMask(poppy.AnalyticOpticalElement):
    """
    My version of an idealized scalar vortex phase mask of a given topological charge -
    the focal-plane optic in a vortex coronagraph, replacing the classical
    Lyot occulter. I settled for Phase-only: transmission is 1 everywhere (nothing is
    blocked by amplitude), and the OPD is chosen so the phase equals
    charge * azimuthal_angle, i.e. the field picks up exp(i*charge*theta).
    """
    def __init__(self, charge=2, **kwargs):
        kwargs.setdefault('planetype', poppy.poppy_core.PlaneType.image)
        super().__init__(**kwargs)
        self.charge = charge
 
    def get_transmission(self, wave):
        y, x = self.get_coordinates(wave)
        return np.ones_like(x)
 
    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        theta = np.arctan2(y, x)
        wl = wave.wavelength.to(u.m).value if hasattr(wave.wavelength, 'to') else wave.wavelength
        return self.charge * theta * wl / (2 * np.pi)
 

npix, oversample = SEARCH_NPIX, SEARCH_OVERSAMPLE
_design_cache = {}
 
 
def set_resolution(new_npix, new_oversample):
    """Switch resolution and clear the design cache """
    global npix, oversample
    npix, oversample = new_npix, new_oversample
    _design_cache.clear()
 
 
def lambda_over_d_to_arcsec(wavelength, diameter):
    theta_rad = (wavelength / diameter) * u.radian
    return theta_rad.to(u.arcsec, equivalencies=u.dimensionless_angles())
 
 
ONE_LD_ARCSEC = lambda_over_d_to_arcsec(wavelength, telescope_diameter).value
 
 

# ---- Optical system builders ---- 
 
def build_lyot_coronagraph(occulter_radius_lD, lyot_stop_fraction):
    pupil_radius = telescope_diameter / 2.0
    occulter_radius_arcsec = occulter_radius_lD * ONE_LD_ARCSEC
    lyot_stop_radius = lyot_stop_fraction * pupil_radius
 
    osys = poppy.OpticalSystem(npix=npix, oversample=oversample)
    osys.add_pupil(poppy.CircularAperture(radius=pupil_radius.to(u.m).value))
    osys.add_image(poppy.CircularOcculter(radius=occulter_radius_arcsec))
    osys.add_pupil(poppy.CircularAperture(radius=lyot_stop_radius.to(u.m).value))
 
    osys.add_detector(pixelscale=NATIVE_PIXELSCALE_ARCSEC, fov_pixels=FOV_PIXELS)
    actual_pixelscale = NATIVE_PIXELSCALE_ARCSEC / oversample
    return osys, actual_pixelscale
 
 
def build_reference_system():
    pupil_radius = telescope_diameter / 2.0
    osys = poppy.OpticalSystem(npix=npix, oversample=oversample)
    osys.add_pupil(poppy.CircularAperture(radius=pupil_radius.to(u.m).value))
    osys.add_detector(pixelscale=NATIVE_PIXELSCALE_ARCSEC, fov_pixels=FOV_PIXELS)
    actual_pixelscale = NATIVE_PIXELSCALE_ARCSEC / oversample
    return osys, actual_pixelscale
 
 
def build_vortex_coronagraph(charge, lyot_stop_fraction):
    """Same four-plane architecture as the lyot, but this plane is
    is a VortexPhaseMask instead of a CircularOcculter - everything else
    (pupil, Lyot stop, detector) is the same"""
    pupil_radius = telescope_diameter / 2.0
    lyot_stop_radius = lyot_stop_fraction * pupil_radius
 
    osys = poppy.OpticalSystem(npix=npix, oversample=oversample)
    osys.add_pupil(poppy.CircularAperture(radius=pupil_radius.to(u.m).value))
    osys.add_image(VortexPhaseMask(charge=charge))
    osys.add_pupil(poppy.CircularAperture(radius=lyot_stop_radius.to(u.m).value))
 
    osys.add_detector(pixelscale=NATIVE_PIXELSCALE_ARCSEC, fov_pixels=FOV_PIXELS)
    actual_pixelscale = NATIVE_PIXELSCALE_ARCSEC / oversample
    return osys, actual_pixelscale
 

# ---- Helpers go here ---
 
def locate_peak_pixel(image):
    iy, ix = np.unravel_index(np.argmax(image), image.shape)
    return (float(ix), float(iy))
 
 
def aperture_sum(image, pixelscale_arcsec, center_xy_pix, radius_arcsec):
    y, x = np.indices(image.shape)
    cx, cy = center_xy_pix
    r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = r_pix * pixelscale_arcsec <= radius_arcsec
    return image[mask].sum()
 
 
def radial_coordinates_lD(shape, pixelscale_arcsec):
    ny, nx = shape
    y, x = np.indices(shape)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_arcsec = r_pix * pixelscale_arcsec
    return r_arcsec / ONE_LD_ARCSEC
 
 
def find_iwa(separations_lD, throughput, threshold=0.5):
    for i in range(len(separations_lD) - 1):
        t0, t1 = throughput[i], throughput[i + 1]
        if np.isnan(t0) or np.isnan(t1):
            continue
        if t0 < threshold <= t1:
            s0, s1 = separations_lD[i], separations_lD[i + 1]
            frac = (threshold - t0) / (t1 - t0)
            return s0 + frac * (s1 - s0)
    return np.nan
 
 
# --- Corner for the reference data ----
 
def precompute_reference_data():
    ref_osys, ref_pixelscale = build_reference_system()
    psf_onaxis = ref_osys.calc_psf(wavelength=wavelength.to(u.m).value)
    unocculted_peak = psf_onaxis[0].data.max()
 
    ref_flux = np.zeros(len(SEPARATIONS_LD))
    ref_centers = []
    ap_radius_arcsec = APERTURE_RADIUS_LD * ONE_LD_ARCSEC
 
    for i, sep_lD in enumerate(SEPARATIONS_LD):
        osys, pixelscale = build_reference_system()
        osys.source_offset_r = sep_lD * ONE_LD_ARCSEC
        osys.source_offset_theta = THETA_DEG
        psf = osys.calc_psf(wavelength=wavelength.to(u.m).value)
        img = psf[0].data
        center = locate_peak_pixel(img)
        ref_centers.append(center)
        ref_flux[i] = aperture_sum(img, pixelscale, center, ap_radius_arcsec)
 
    return dict(unocculted_peak=unocculted_peak, ref_flux=ref_flux,
                ref_centers=ref_centers, pixelscale=ref_pixelscale)
 

# ---- Evaluating one design -----


def evaluate_design(occulter_radius_lD, lyot_stop_fraction, ref_data):
    key = (round(occulter_radius_lD, 4), round(lyot_stop_fraction, 4))
    if key in _design_cache:
        return _design_cache[key]
 
    ap_radius_arcsec = APERTURE_RADIUS_LD * ONE_LD_ARCSEC
    corona_osys, pixelscale = build_lyot_coronagraph(occulter_radius_lD, lyot_stop_fraction)
 
    throughput = np.full(len(SEPARATIONS_LD), np.nan)
    for i, sep_lD in enumerate(SEPARATIONS_LD):
        osys, _ = build_lyot_coronagraph(occulter_radius_lD, lyot_stop_fraction)
        osys.source_offset_r = sep_lD * ONE_LD_ARCSEC
        osys.source_offset_theta = THETA_DEG
        psf = osys.calc_psf(wavelength=wavelength.to(u.m).value)
        img = psf[0].data
        center = ref_data["ref_centers"][i]
        flux = aperture_sum(img, pixelscale, center, ap_radius_arcsec)
        throughput[i] = flux / ref_data["ref_flux"][i] if ref_data["ref_flux"][i] > 0 else np.nan
 
    iwa = find_iwa(SEPARATIONS_LD, throughput, threshold=0.5)
 
    if np.isnan(iwa) or (1.5 * iwa) > SEPARATIONS_LD[-1]:
        result = dict(occulter_radius_lD=occulter_radius_lD, lyot_stop_fraction=lyot_stop_fraction,
                       iwa=iwa, contrast=1.0, throughput_ref=0.0, valid=False)
        _design_cache[key] = result
        return result
 
    throughput_ref = float(np.interp(1.5 * iwa, SEPARATIONS_LD, throughput))
 
    psf_onaxis = corona_osys.calc_psf(wavelength=wavelength.to(u.m).value)
    corona_image = psf_onaxis[0].data
    r_lD = radial_coordinates_lD(corona_image.shape, pixelscale)
 
    if iwa < OWA_LD:
        bin_edges = np.linspace(iwa, OWA_LD, N_CONTRAST_BINS + 1)
        bin_contrasts = []
        for i in range(N_CONTRAST_BINS):
            mask = (r_lD >= bin_edges[i]) & (r_lD < bin_edges[i + 1])
            if np.any(mask):
                bin_contrasts.append(corona_image[mask].mean() / ref_data["unocculted_peak"])
        contrast = max(bin_contrasts) if bin_contrasts else 1.0
    else:
        contrast = 1.0
 
    result = dict(occulter_radius_lD=occulter_radius_lD, lyot_stop_fraction=lyot_stop_fraction,
                   iwa=iwa, contrast=float(contrast), throughput_ref=throughput_ref, valid=True)
    _design_cache[key] = result
    return result
 
 
def presample_metric_ranges(ref_data, n_samples=15, seed=0):
    rng = np.random.default_rng(seed)
    occulter_samples = rng.uniform(BOUNDS[0][0], BOUNDS[0][1], n_samples)
    lyot_samples = rng.uniform(BOUNDS[1][0], BOUNDS[1][1], n_samples)
    log_contrasts, throughputs = [], []
    for occ, lyot in zip(occulter_samples, lyot_samples):
        m = evaluate_design(occ, lyot, ref_data)
        if m["valid"]:
            log_contrasts.append(np.log10(max(m["contrast"], 1e-12)))
            throughputs.append(m["throughput_ref"])
    if not log_contrasts:
        return dict(log_c_min=-6.0, log_c_max=-3.0, t_min=0.0, t_max=1.0)
    return dict(log_c_min=min(log_contrasts), log_c_max=max(log_contrasts),
                t_min=min(throughputs), t_max=max(throughputs))
 
 

# ---- Timing calibration (this has to go before the real sweep so this old dino can make an estimate ----

def estimate_sweep_runtime(ref_data):
    t0 = time.time()
    evaluate_design(4.5, 0.85, ref_data)  # a representative, uncached point
    single_eval_s = time.time() - t0
 
    approx_evals_per_epsilon = DE_POPSIZE * (DE_MAXITER + 1)
    total_s = single_eval_s * approx_evals_per_epsilon * N_EPSILON_STEPS
 
    print(f"\nTiming calibration: one design eval took {single_eval_s:.3f} s "
          f"at npix={npix}, oversample={oversample}.")
    print(f"Rough estimate for the full search sweep: "
          f"~{approx_evals_per_epsilon} evals/epsilon x {N_EPSILON_STEPS} epsilons "
          f"x {single_eval_s:.3f} s = ~{total_s/60:.1f} minutes.")
    
    return total_s
 
 

# ---- Epsilon-constraint sweep ----

def make_contrast_objective(ref_data):
    def objective(x):
        m = evaluate_design(x[0], x[1], ref_data)
        return np.log10(max(m["contrast"], 1e-12)) if m["valid"] else 5.0
    return objective
 
 
def make_throughput_constraint(ref_data, epsilon):
    def constraint_fn(x):
        m = evaluate_design(x[0], x[1], ref_data)
        return (m["throughput_ref"] - epsilon) if m["valid"] else -1.0
    return constraint_fn
 
 
def make_iwa_constraint(ref_data, iwa_max):
    """Positive means satisfied: iwa_max - IWA >= 0, i.e. IWA <= iwa_max."""
    def constraint_fn(x):
        m = evaluate_design(x[0], x[1], ref_data)
        if not m["valid"] or np.isnan(m["iwa"]):
            return -1.0
        return iwa_max - m["iwa"]
    return constraint_fn
 
 
def run_epsilon_sweep(ref_data, epsilons):
    objective = make_contrast_objective(ref_data)
    pareto_points = []
    for eps in epsilons:
        t_start = time.time()
        print(f"\nOptimizing at throughput floor epsilon={eps:.3f}...")
        constraint_fn = make_throughput_constraint(ref_data, eps)
        iwa_constraint_fn = make_iwa_constraint(ref_data, IWA_MAX_LD)
        nlc_throughput = NonlinearConstraint(constraint_fn, 0, np.inf)
        nlc_iwa = NonlinearConstraint(iwa_constraint_fn, 0, np.inf)
        result = differential_evolution(
            objective, BOUNDS, constraints=(nlc_throughput, nlc_iwa),
            popsize=DE_POPSIZE, maxiter=DE_MAXITER,
            seed=0, polish=False, tol=1e-3,
        )
        occulter_radius_lD, lyot_stop_fraction = result.x
        cached_metrics = evaluate_design(occulter_radius_lD, lyot_stop_fraction, ref_data)
        metrics = dict(cached_metrics)
        metrics["epsilon"] = eps
        iwa_ok = metrics["valid"] and not np.isnan(metrics["iwa"]) and metrics["iwa"] <= IWA_MAX_LD + 1e-3
        throughput_ok = metrics["valid"] and (metrics["throughput_ref"] >= eps - 1e-3)
        metrics["constraint_satisfied"] = iwa_ok and throughput_ok
        pareto_points.append(metrics)
 
        flags = []
        if not throughput_ok:
            flags.append("throughput floor not met")
        if not iwa_ok:
            flags.append(f"IWA={metrics['iwa']:.2f} exceeds cap of {IWA_MAX_LD}")
        flag = f"  [{'; '.join(flags)}]" if flags else ""
        print(f"  best: occulter={occulter_radius_lD:.3f} lambda/D, "
              f"Lyot stop={lyot_stop_fraction:.3f}, IWA={metrics['iwa']:.2f} lambda/D, "
              f"contrast={metrics['contrast']:.3e}, throughput={metrics['throughput_ref']:.3f}"
              f"{flag}  ({time.time()-t_start:.1f} s)")
 
        occ_at_bound = np.isclose(occulter_radius_lD, BOUNDS[0][1], atol=0.02)
        lyot_at_bound = np.isclose(lyot_stop_fraction, BOUNDS[1][1], atol=0.005) or \
                        np.isclose(lyot_stop_fraction, BOUNDS[1][0], atol=0.005)
        if occ_at_bound or lyot_at_bound:
            print("  [WARNING: the result is sitting at a search bound. Gonna have to widen them]")
    return pareto_points
 
 
def filter_dominated(points):
    """
    Keeping only the non-dominated subset
    """
    non_dominated, dominated = [], []
    for p in points:
        is_dominated = any(
            (q["throughput_ref"] >= p["throughput_ref"] and q["contrast"] <= p["contrast"]
             and (q["throughput_ref"] > p["throughput_ref"] or q["contrast"] < p["contrast"]))
            for q in points if q is not p
        )
        (dominated if is_dominated else non_dominated).append(p)
    return non_dominated, dominated
 
 
def verify_at_full_resolution(pareto_points):
    """
    Re-evaluating each search-phase winner at full res. One forward pass per point
    """
    print(f"\nVerifying {len(pareto_points)} winning designs at full resolution"
          f"(npix={VERIFY_NPIX}, oversample={VERIFY_OVERSAMPLE})")
    set_resolution(VERIFY_NPIX, VERIFY_OVERSAMPLE)
    verify_ref_data = precompute_reference_data()
 
    verified = []
    for p in pareto_points:
        
        m = dict(evaluate_design(p["occulter_radius_lD"], p["lyot_stop_fraction"], verify_ref_data))
        m["epsilon"] = p["epsilon"]
        verified.append(m)
        d_contrast = (m["contrast"] - p["contrast"]) / p["contrast"] * 100 if p["contrast"] else float("nan")
        d_through = (m["throughput_ref"] - p["throughput_ref"]) * 100
        print(f"  epsilon={p['epsilon']:.3f}: search contrast={p['contrast']:.3e} -> "
              f"verify {m['contrast']:.3e} ({d_contrast:+.1f}%); "
              f"search throughput={p['throughput_ref']:.3f} -> "
              f"verify {m['throughput_ref']:.3f} ({d_through:+.1f} pts)")
    return verified
 
 

# ---- Vortex vs Lyot time: single-point comparison ---
 
def evaluate_vortex_design(charge, lyot_stop_fraction, ref_data):

    ap_radius_arcsec = APERTURE_RADIUS_LD * ONE_LD_ARCSEC
    corona_osys, pixelscale = build_vortex_coronagraph(charge, lyot_stop_fraction)
 
    throughput = np.full(len(SEPARATIONS_LD), np.nan)
    for i, sep_lD in enumerate(SEPARATIONS_LD):
        osys, _ = build_vortex_coronagraph(charge, lyot_stop_fraction)
        osys.source_offset_r = sep_lD * ONE_LD_ARCSEC
        osys.source_offset_theta = THETA_DEG
        psf = osys.calc_psf(wavelength=wavelength.to(u.m).value)
        img = psf[0].data
        center = ref_data["ref_centers"][i]
        flux = aperture_sum(img, pixelscale, center, ap_radius_arcsec)
        throughput[i] = flux / ref_data["ref_flux"][i] if ref_data["ref_flux"][i] > 0 else np.nan
 
    iwa = find_iwa(SEPARATIONS_LD, throughput, threshold=0.5)
    result = dict(charge=charge, lyot_stop_fraction=lyot_stop_fraction, iwa=iwa)
 
    if np.isnan(iwa) or (1.5 * iwa) > SEPARATIONS_LD[-1]:
        result.update(contrast=1.0, throughput_ref=0.0, valid=False)
        return result
 
    throughput_ref = float(np.interp(1.5 * iwa, SEPARATIONS_LD, throughput))
 
    psf_onaxis = corona_osys.calc_psf(wavelength=wavelength.to(u.m).value)
    corona_image = psf_onaxis[0].data
    r_lD = radial_coordinates_lD(corona_image.shape, pixelscale)
 
    if iwa < OWA_LD:
        bin_edges = np.linspace(iwa, OWA_LD, N_CONTRAST_BINS + 1)
        bin_contrasts = []
        for i in range(N_CONTRAST_BINS):
            mask = (r_lD >= bin_edges[i]) & (r_lD < bin_edges[i + 1])
            if np.any(mask):
                bin_contrasts.append(corona_image[mask].mean() / ref_data["unocculted_peak"])
        contrast = max(bin_contrasts) if bin_contrasts else 1.0
    else:
        contrast = 1.0
 
    result.update(contrast=float(contrast), throughput_ref=throughput_ref, valid=True)
    return result
 
 
def run_vortex_vs_lyot_comparison(best_lyot_point):
    """
    Single-point, resolution-matched comparison: I'm taking the Lyot stop
    fraction from one representative classical-Lyot design, and
    evaluating both the classical occulter and the charge-2 vortex at that
    same Lyot stop, using a single high res. Also printting the vortex's own sampling-sensitivity table
    """
    lyot_stop_fraction = best_lyot_point["lyot_stop_fraction"]
    print(f"\n--- Vortex (charge={VORTEX_CHARGE}) vs classical Lyot, "
          f"matched Lyot stop={lyot_stop_fraction:.3f}, "
          f"at npix={VORTEX_NPIX}, oversample={VORTEX_OVERSAMPLE} ---")
 
    set_resolution(VORTEX_NPIX, VORTEX_OVERSAMPLE)
    matched_ref_data = precompute_reference_data()
 
    lyot_matched = evaluate_design(best_lyot_point["occulter_radius_lD"], lyot_stop_fraction, matched_ref_data)
    vortex_matched = evaluate_vortex_design(VORTEX_CHARGE, lyot_stop_fraction, matched_ref_data)
 
    print(f"  classical Lyot  (occulter={best_lyot_point['occulter_radius_lD']:.2f} lambda/D): "
          f"IWA={lyot_matched['iwa']:.2f}, contrast={lyot_matched['contrast']:.3e}, "
          f"throughput={lyot_matched['throughput_ref']:.3f}")
    print(f"  vortex (charge={VORTEX_CHARGE}):                        "
          f"IWA={vortex_matched['iwa']:.2f}, contrast={vortex_matched['contrast']:.3e}, "
          f"throughput={vortex_matched['throughput_ref']:.3f}")
 
    print("\n  Vortex null-depth sampling sensitivity (same design, resolution varied only):")
    for test_npix, test_oversample in [(128, 2), (256, 4), (256, 8)]:
        set_resolution(test_npix, test_oversample)
        test_ref = precompute_reference_data()
        m = evaluate_vortex_design(VORTEX_CHARGE, lyot_stop_fraction, test_ref)
        print(f"    npix={test_npix}, oversample={test_oversample}: contrast={m['contrast']:.3e}")
 
    # restore verify resolution for anything downstream
    set_resolution(VERIFY_NPIX, VERIFY_OVERSAMPLE)
    return lyot_matched, vortex_matched
 

# ---- Main block ----
 
if __name__ == "__main__":
 
    set_resolution(SEARCH_NPIX, SEARCH_OVERSAMPLE)
 
    print(f"Search phase resolution: npix={npix}, oversample={oversample}")
    print("Precomputing reference (bare aperture) data...")
    ref_data = precompute_reference_data()
 
    estimate_sweep_runtime(ref_data)
 
    print("\nPresampling metric ranges to pick epsilon (throughput floor) targets")
    norm = presample_metric_ranges(ref_data, n_samples=15, seed=0)
    print(f"  log10(contrast) range: [{norm['log_c_min']:.2f}, {norm['log_c_max']:.2f}]")
    print(f"  throughput range:      [{norm['t_min']:.3f}, {norm['t_max']:.3f}]")
 
    pad = 0.02 * max(norm["t_max"] - norm["t_min"], 1e-3)
    EPSILONS = np.linspace(norm["t_min"] + pad, norm["t_max"] - pad, N_EPSILON_STEPS)
    print(f"  epsilon targets: {np.round(EPSILONS, 3)}")
 
    t_sweep_start = time.time()
    pareto_points = run_epsilon_sweep(ref_data, EPSILONS)
    print(f"\nSearch sweep finished in {(time.time()-t_sweep_start)/60:.1f} minutes.")
 
    verified_points = verify_at_full_resolution(pareto_points)
    valid_points_all = [p for p in verified_points if p["valid"]]
    infeasible_points = [p for p in verified_points if not p["valid"]]
 
    valid_points, dominated_points = filter_dominated(valid_points_all)
    if dominated_points:
        print(f"\n{len(dominated_points)} of {len(valid_points_all)} valid points were "
        )
        for p in dominated_points:
            print(f"  epsilon={p['epsilon']:.3f}: occulter={p['occulter_radius_lD']:.3f}, "
                  f"contrast={p['contrast']:.3e}, throughput={p['throughput_ref']:.3f} "
            )
 
    print("\n Verified (full-res) Pareto front summary")
    print(f"{'epsilon':>8} {'occulter':>9} {'lyot':>6} {'IWA':>6} {'contrast':>11} {'throughput':>11}")
    for p in verified_points:
        print(f"{p['epsilon']:8.3f} {p['occulter_radius_lD']:9.3f} {p['lyot_stop_fraction']:6.3f} "
              f"{p['iwa']:6.2f} {p['contrast']:11.3e} {p['throughput_ref']:11.3f}")
 

    # Picking a representative verified point

    representative_point = sorted(valid_points, key=lambda p: p["epsilon"])[len(valid_points) // 2]
    lyot_matched, vortex_matched = run_vortex_vs_lyot_comparison(representative_point)
 
    fig, ax = plt.subplots(figsize=(6, 5))
    contrasts = [p["contrast"] for p in valid_points]
    throughputs = [p["throughput_ref"] for p in valid_points]
    epsilons_plot = [p["epsilon"] for p in valid_points]
 
    sc = ax.scatter(throughputs, contrasts, c=epsilons_plot, cmap="viridis", s=80, zorder=3)
    order = np.argsort(throughputs)
    ax.plot(np.array(throughputs)[order], np.array(contrasts)[order], "--", color="gray", lw=1, zorder=2)
    for p in valid_points:
        ax.annotate(f"occ={p['occulter_radius_lD']:.1f}\nlyot={p['lyot_stop_fraction']:.2f}",
                     (p["throughput_ref"], p["contrast"]), fontsize=7,
                     textcoords="offset points", xytext=(6, 6))
    if infeasible_points:
        ax.scatter([p["throughput_ref"] for p in infeasible_points],
                   [p["contrast"] for p in infeasible_points],
                   marker="x", color="red", s=60, zorder=4, label="infeasible")
    if dominated_points:
        ax.scatter([p["throughput_ref"] for p in dominated_points],
                   [p["contrast"] for p in dominated_points],
                   marker="o", facecolors="none", edgecolors="gray", s=100, zorder=4,
                   label="dominated (excluded from front)")
 
    # Overlaying the single-point vortex vs Lyot comparison, at matched Lyot stop and matched res

    if vortex_matched["valid"]:
        ax.scatter([vortex_matched["throughput_ref"]], [vortex_matched["contrast"]],
                    marker="*", color="darkorange", s=220, zorder=5,
                    label=f"vortex (charge={VORTEX_CHARGE}), matched Lyot stop")
    ax.scatter([lyot_matched["throughput_ref"]], [lyot_matched["contrast"]],
                marker="P", color="black", s=100, zorder=5,
                label="classical Lyot, same comparison (matched resolution)")
    ax.legend(fontsize=8)
 
    ax.set_yscale("log")
    ax.set_xlabel("Throughput (at 1.5 x IWA)")
    ax.set_ylabel(f"Contrast (worst annulus, IWA to {OWA_LD:.0f} lambda/D)")
    ax.set_title("Epsilon-constraint Pareto front (full resolution)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("epsilon (throughput floor)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_front_verified.png", dpi=150)
    print("\nSaved plot as pareto_front_verified.png")
    plt.show()
 
    print("Done, dude!")