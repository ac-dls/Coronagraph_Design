# Tier 1 of extra results produced form the Lyot vs Vortex coronagraph machine
 
import numpy as np
import matplotlib.pyplot as plt
import poppy
from astropy import units as u
import Lyot_coronagraph_fast as core
 

# ---- Representative design from the previous code ----

REPRESENTATIVE_OCCULTER_LD = 3.925
REPRESENTATIVE_LYOT_STOP = 0.887
 

# ---- Angular "reality check" ----

def print_angular_reality_check():
    print("=" * 78)
    print("Converting verified IWA (in lambda/D) to arcsec")
 
    # Also From the last run (epsilon=0.700 row + matched vortex point)
    classical_iwa_ld = 4.98
    vortex_iwa_ld = 1.94
 
    configs = [
        ("This simulator's own setup", 6.0 * u.m, 1.0 * u.micron,
         "(consistency check against the numbers already produced above)"),
        ("Space-telescope-like, visible", 6.0 * u.m, 0.5 * u.micron,
         "(HWO-class aperture, visible-light imaging of an Earth analog)"),
        ("Real VLT/NACO AGPM setup", 8.0 * u.m, 3.8 * u.micron,
         "(L-band ground-based instrument used in the published"
         " vortex/beta Pic results)"),
    ]
 
    print(f"\n{'Configuration':<32}{'1 lambda/D':>12}{'Classical IWA':>16}{'Vortex IWA':>14}")
    for label, diam, wl, note in configs:
        one_ld_arcsec = core.lambda_over_d_to_arcsec(wl, diam).value
        classical_arcsec = classical_iwa_ld * one_ld_arcsec
        vortex_arcsec = vortex_iwa_ld * one_ld_arcsec
        print(f"{label:<32}{one_ld_arcsec*1000:>9.1f} mas{classical_arcsec*1000:>13.1f} mas"
              f"{vortex_arcsec*1000:>11.1f} mas")
        print(f"    {note}")
 
    print("\nLiterature benchmarks for comparison:")
    print("  beta Pic (VLT/NACO AGPM, L-band, achieved):        ~100 mas (~0.1 arcsec)")
    print("  Alpha Cen A habitable zone (~1 AU at 1.34 pc):     ~750 mas (~0.75 arcsec)")
    print()
 
 
# ---- Unaberrated stellar PSF plot ----
 
def plot_unaberrated_psf():
    core.set_resolution(core.VERIFY_NPIX, core.VERIFY_OVERSAMPLE)
    osys, pixelscale = core.build_reference_system()
    psf = osys.calc_psf(wavelength=core.wavelength.to(u.m).value)
 
    fig, ax = plt.subplots(figsize=(5, 5))
    image = psf[0].data
    display = np.log10(image / image.max() + 1e-12)
    half_width = pixelscale * image.shape[0] / 2.0
    im = ax.imshow(display, origin="lower", cmap="inferno", vmin=-8, vmax=0,
                   extent=[-half_width, half_width, -half_width, half_width])
    ax.set_title("Unaberrated stellar PSF (bare aperture, no coronagraph)")
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    fig.colorbar(im, ax=ax, label="log10(intensity / peak)")
    plt.tight_layout()
    plt.savefig("tier1_unaberrated_psf.png", dpi=150)
    print("Saved tier1_unaberrated_psf.png")
    plt.show()
 
 
#---- Plane-by-plane fields: classical Lyot (top row) vs vortex (bottom row) ----
 
def _plot_planes_row(axes_row, osys, wavelength_m, plane_labels):
    psf, intermediates = osys.calc_psf(wavelength=wavelength_m, return_intermediates=True)
    for i, wf in enumerate(intermediates):
        intensity = wf.intensity
        display = np.log10(intensity / intensity.max() + 1e-12)
        pixscale = wf.pixelscale.value if hasattr(wf.pixelscale, "value") else wf.pixelscale
        half_width = pixscale * intensity.shape[0] / 2.0
        is_pupil = (wf.planetype == poppy.poppy_core.PlaneType.pupil)
        unit_label = "meters" if is_pupil else "arcsec"
        axes_row[i].imshow(display, origin="lower", cmap="inferno", vmin=-8, vmax=0,
                            extent=[-half_width, half_width, -half_width, half_width])
        axes_row[i].set_title(plane_labels[i], fontsize=9)
        axes_row[i].set_xlabel(unit_label, fontsize=7)
        axes_row[i].tick_params(labelsize=6)
 
 
def plot_lyot_vs_vortex_planes(occulter_lD, lyot_stop_fraction, charge):
    core.set_resolution(core.VERIFY_NPIX, core.VERIFY_OVERSAMPLE)
    lyot_osys, _ = core.build_lyot_coronagraph(occulter_lD, lyot_stop_fraction)
    wl_m = core.wavelength.to(u.m).value
 
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    _plot_planes_row(axes[0], lyot_osys, wl_m,
                      ["A: Entrance pupil", "B: Occulted focal plane",
                       "C: Lyot-stop output (post-clip pupil)", "D: Final image"])
 
    # Vortex needs the finer resolution established during development to
    # reach anywhere near its real null depth -- same reasoning as the
    # main comparison in Lyot_coronagraph_fast.py.
    core.set_resolution(core.VORTEX_NPIX, core.VORTEX_OVERSAMPLE)
    vortex_osys, _ = core.build_vortex_coronagraph(charge, lyot_stop_fraction)
    _plot_planes_row(axes[1], vortex_osys, wl_m,
                      [f"A: Entrance pupil", f"B: Vortex phase-mask field (charge={charge})",
                       "C: Lyot-stop output (post-clip pupil)", "D: Final image"])
 
    fig.suptitle(f"Plane-by-plane comparison at matched Lyot stop = {lyot_stop_fraction:.3f}\n"
                 f"Top: classical Lyot (occulter={occulter_lD:.2f} lambda/D)   "
                 f"Bottom: vortex (charge={charge})", fontsize=11)
    plt.tight_layout()
    plt.savefig("tier1_lyot_vs_vortex_planes.png", dpi=150)
    print("Saved tier1_lyot_vs_vortex_planes.png")
    plt.show()
 
    # Restoring verify resolution for anything downstream
    core.set_resolution(core.VERIFY_NPIX, core.VERIFY_OVERSAMPLE)
 

# ---- Full throughput-vs-separation and contrast-vs-separation curves for the comparison ----
 
def compute_full_curves(build_fn, build_args, npix, oversample, ref_data):
   
    core.set_resolution(npix, oversample)
    osys, pixelscale = build_fn(*build_args)
    ap_radius_arcsec = core.APERTURE_RADIUS_LD * core.ONE_LD_ARCSEC
 
    throughput = np.full(len(core.SEPARATIONS_LD), np.nan)
    for i, sep_lD in enumerate(core.SEPARATIONS_LD):
        o, _ = build_fn(*build_args)
        o.source_offset_r = sep_lD * core.ONE_LD_ARCSEC
        o.source_offset_theta = core.THETA_DEG
        psf = o.calc_psf(wavelength=core.wavelength.to(u.m).value)
        img = psf[0].data
        center = ref_data["ref_centers"][i]
        flux = core.aperture_sum(img, pixelscale, center, ap_radius_arcsec)
        throughput[i] = flux / ref_data["ref_flux"][i] if ref_data["ref_flux"][i] > 0 else np.nan
 
    psf_onaxis = osys.calc_psf(wavelength=core.wavelength.to(u.m).value)
    corona_image = psf_onaxis[0].data
    r_lD = core.radial_coordinates_lD(corona_image.shape, pixelscale)
 
    bin_edges = np.linspace(0.5, core.OWA_LD, 40)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    contrast_curve = np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        mask = (r_lD >= bin_edges[i]) & (r_lD < bin_edges[i + 1])
        if np.any(mask):
            contrast_curve[i] = corona_image[mask].mean() / ref_data["unocculted_peak"]
 
    return core.SEPARATIONS_LD, throughput, bin_centers, contrast_curve
 
 
def plot_throughput_and_contrast_curves(occulter_lD, lyot_stop_fraction, charge):
    core.set_resolution(core.VERIFY_NPIX, core.VERIFY_OVERSAMPLE)
    lyot_ref_data = core.precompute_reference_data()
    seps, lyot_throughput, bins, lyot_contrast = compute_full_curves(
        core.build_lyot_coronagraph, (occulter_lD, lyot_stop_fraction),
        core.VERIFY_NPIX, core.VERIFY_OVERSAMPLE, lyot_ref_data)
 
    core.set_resolution(core.VORTEX_NPIX, core.VORTEX_OVERSAMPLE)
    vortex_ref_data = core.precompute_reference_data()
    _, vortex_throughput, _, vortex_contrast = compute_full_curves(
        core.build_vortex_coronagraph, (charge, lyot_stop_fraction),
        core.VORTEX_NPIX, core.VORTEX_OVERSAMPLE, vortex_ref_data)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
 
    ax1.plot(seps, lyot_throughput, "o-", label=f"classical Lyot (occulter={occulter_lD:.2f} lambda/D)")
    ax1.plot(seps, vortex_throughput, "s-", label=f"vortex (charge={charge})")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, label="50% (IWA definition)")
    ax1.set_xlabel("Separation (lambda/D)")
    ax1.set_ylabel("Throughput (relative to bare aperture)")
    ax1.set_title("Off-axis planet throughput vs. separation")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
 
    ax2.semilogy(bins, lyot_contrast, "o-", ms=4, label="classical Lyot")
    ax2.semilogy(bins, vortex_contrast, "s-", ms=4, label="vortex")
    ax2.set_xlabel("Separation (lambda/D)")
    ax2.set_ylabel("Contrast (relative to unocculted peak)")
    ax2.set_title("Contrast vs. separation (lambda/D and arcsec)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
 
    # Secondary x-axis in arcsec, at this simulator's own wavelength/aperture

    def ld_to_arcsec(x):
        return x * core.ONE_LD_ARCSEC
    def arcsec_to_ld(x):
        return x / core.ONE_LD_ARCSEC
    secax = ax2.secondary_xaxis('top', functions=(ld_to_arcsec, arcsec_to_ld))
    secax.set_xlabel("Separation (arcsec)")
 
    plt.tight_layout()
    plt.savefig("tier1_throughput_contrast_curves.png", dpi=150)
    print("Saved tier1_throughput_contrast_curves.png")
    plt.show()
 
    core.set_resolution(core.VERIFY_NPIX, core.VERIFY_OVERSAMPLE)
 
 
if __name__ == "__main__":
    print_angular_reality_check()
    plot_unaberrated_psf()
    plot_lyot_vs_vortex_planes(REPRESENTATIVE_OCCULTER_LD, REPRESENTATIVE_LYOT_STOP, core.VORTEX_CHARGE)
    plot_throughput_and_contrast_curves(REPRESENTATIVE_OCCULTER_LD, REPRESENTATIVE_LYOT_STOP, core.VORTEX_CHARGE)
    print("Done, dude!")