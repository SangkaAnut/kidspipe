# Provenance of the official KiDS-Legacy xi+- / COSEBIs TreeCorr settings (verifier: provenance)

## Verdict on C4
"Official was measured with bin_slop 1.5" is DOCUMENTED (not merely "behaves like"). Wright+25 (arXiv:2503.19441v2,
A&A 703 A158) Sect. 2.5.1, footnote 1 (p.6): "We use Treecorr version 4.2.3, with binslop = 1.5."
Same paper Sect. 2.5.1: "we measure xi+-(theta) for 1000 logarithmically spaced theta-bins between theta_min = 2' and
theta_max = 300' using Treecorr"; Sect. 2.5 intro: COSEBIs/band powers are "linear combinations of finely-binned 2PCFs" (Eq. 16);
Appendix F Eq. F.1: 9-bin xi = N_pair-weighted rebin of the 1000 fine bins (N_pair = column npairs_weighted).
KiDS-1000 precedent: Joachimi+21 (arXiv:2007.01844) Sect. 3.5: "fairly large values of bin_slop (1.5 for cosmic shear and 1.2 for GGL)
can be used" because of >100 bins per decade; coarse bins are "obtained by merging the finely binned measurements".

NOT documented anywhere I could find: angle_slop (does not exist in 4.2.3), ordered vs unordered, metric (sphere/Arc vs chordal default),
patches/low-level options, any bin-weights, number of threads. The sub-claim "the A3 xi- failure reflects the official release's bin_slop
approximation, not an error in our measurement" is an INFERENCE, supported by (a) the documented bin_slop 1.5 and (b) the
bin_slop-1.5-through-5.1.4 run matching official to 0.044 sigma vs exact 0.157 (PILOT_BIN6.md item 2), but no source states the
official xi was "approximate"; a residual alternative (official 4.2.3 vs 5.1.4 numerics, or different catalogue realisation/m handling)
is not excluded by documentation. See caveats.

## Official data products (headers, read myself)
Dir: /mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/
- data/KiDS_Legacy_xipm.fits, _cosebis.fits, _bandpowers.fits: ALL header cards printed. Only generic 2pt-FITS keywords
  (COVDATA, 2PTDATA, EXTNAME, QUANT1/2 = G+R/G-R/GEF/GBF, KERNEL_1/2=NZ_SOURCE, WINDOWS=SAMPLE, N_ZBIN_1/2=6, N_ANG = 9 / 20 / 8, STRT_/NAME_)
  and NZ_SOURCE SIG_E_1..6, NGAL_1..6. NO TreeCorr version, bin_slop, angle_slop, nbins, ordering, metric, or rebinning card.
- data/cosmosis_xipm/XI_KiDS_Legacy_NScomb_nBins_6_BinI_BinJ.ascii: single header row
  "r_nom meanr meanlogr xip xim xip_im xim_im sigma_xip sigma_xim npairs weight npairs_weighted"; no settings. (r_nom first value 2.005017 = centre of 1000 log bins 2-300.)
- README.txt (release root): only says "2PCF measurements from treecorr"; grep for slop/binning/nbins finds nothing.
- chains_and_config_files/xipm/KiDS_Legacy_xipm_pipeline.ini lines 46-77: [xip_binned]/[xim_binned] theta_min=2.00 theta_max=300.00 ntheta=9
  weighted_binning=1 column_theta_name=meanr column_npair_name=npairs_weighted nbins_in=6 (this is the 2pt_stats bin_xi module, i.e. the
  fine->9 bin rebin; consistent with Eq. F.1). cosebis ini lines 52-53 use theta_min/max_cosebis only. No TreeCorr settings anywhere in the ini files.
- /mnt/d/HI-gamma-correlation: grep bin_slop|angle_slop|treecorr finds only the prior agent's own pipeline (pipeline/01_lensing_validation,
  03_realspace_xit, agent_prompts) and OneCovariance code; no official-pipeline configuration. /mnt/d/MK-KIDs1000 is KiDS-1000 data, no config found by grep.

## TreeCorr 4.x vs 5.x (angle_slop)
- TreeCorr CHANGELOG "Changes from version 4.3 to 5.0", New features: "Added angle_slop option to separately tune the allowed angular slop
  from using cells, irrespective of the binning." (#170; fetched from github releases/5.0 branch CHANGELOG.rst). So 4.2.3 has NO angle_slop: the
  projection-angle tolerance was tied to the bin-slop split criterion (b = bin_size*bin_slop).
- Installed 5.1.4 docs, corr2base.py:183-190: angle_slop "applies to the projection angle ... will differ by no more than angle_slop radians ...
  (default: 0.1)"; line 274 _default_angle_slop = 0.1; corr2base.py:1606 min cell size uses b1 = min(angle_slop, b).
- My numbers: bin_size = ln(150)/1000 = 5.0106e-3; b(bin_slop 1.5) = 7.516e-3 rad; b(0.1)=5.0e-4; b(0)=0. Inference (NOT verified in C++ source,
  which is compiled; I ran no TreeCorr): in 4.x at bin_slop 1.5 the projection-angle error per cell pair is bounded by ~b ~ 7.5e-3 rad, <= the
  5.x default angle_slop 0.1. In 5.x at bin_slop 1.5 the binding limit is min(angle_slop, b) = b, so the 5.1.4 bin_slop 1.5 run is not made
  looser in angle by the new default; at bin_slop 0 the 4.x angle error was 0 and 5.x with default angle_slop 0.1 is NOT zero (hence the xi- effect
  of RESUME bug 13). bin_slop semantics (allowed error in bin placement in units of bin widths) are unchanged in the docs; I found no changelog
  entry changing them. A numerical 4.2.3-vs-5.1.4 comparison was not possible (no 4.2.3 installed, no pip allowed, no runs).
- Consequence: the official xi was never "exact" in any TreeCorr 4.x sense for bin_slop>0 either; but 4.x cannot be run at (0, angle 0) vs 1.5 in a
  way that corresponds to 5.x angle_slop - the official equals "bin_slop 1.5, angle control implicit via b".

## How "theirs" was configured
Not an external official run: it is the earlier agent's own measurement /mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation
(code/config.py:35-37: comment "Wright+25 Sect. 2.5.1 and footnote 1: TreeCorr v4.2.3, bin_slop=1.5, 1000 log bins, 2'-300'"; XI = dict(min_sep=2.0,
max_sep=300.0, nbins=1000, sep_units='arcmin', bin_type='Log', bin_slop=1.5)). code/xipm.py:41-48: GGCorrelation(**xi, num_threads), auto pairs via
gg.process(cat) (single catalogue => unordered pairs), cross via gg.process(cat_i, cat_j); angle_slop NOT set (5.x default 0.1); metric NOT set
(TreeCorr default for ra/dec); weights = catalogue 'weight'; m applied afterwards as xi_raw/((1+m_i)(1+m_j)). npz xipm_bin6_bin6.npz stores
treecorr_version = 5.1.4 and settings JSON with bin_slop 1.5, runtime 610.6 s. So the "4.2.3" in their report is a citation, the run used 5.1.4
(PRIOR_VALIDATION_01.md s7 notes this). ordered/unordered, metric, angle_slop were the agent's choices, not from the paper. Its npairs are integer
(0 non-integer entries, first 646910), whereas the official ascii npairs is continuous (RESUME headline) - an unexplained difference, so the official
run was not a plain "unweighted cell-count" TreeCorr 4.2.3 call or npairs was post-processed; undocumented.

## Caveats / what could not be checked
- No document states the official ordered/unordered convention, metric, or that angle was approximated; official npairs fractional unexplained.
- No TreeCorr 4.2.3 available; 4.x angle behaviour is from changelog + docs reasoning, not executed or C++-source-verified.
- Asgari+21 (KiDS-1000 cosmic shear, arXiv:2007.15633) not queried; Joachimi+21 Sect. 3.5 already gives the KiDS-1000 setting (bin_slop 1.5).
- Stolzner+25 not queried (not about measurement settings); release README lists it only as a cite.
