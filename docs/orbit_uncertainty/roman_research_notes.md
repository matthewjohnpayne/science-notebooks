> Research notes compiled on 2026-10-18 by a Claude research agent for the planned Roman notebook (scenario 4 in PLAN.md). Bibliographic details were checked against NASA ADS and web sources were read on that date, but treat everything here as notes to verify before quoting in a notebook.

# Roman / GBTDS research notes for a TNO orbit-uncertainty notebook

Compiled 2026-10-18. Bibliographic details were checked against the NASA ADS API (bibcodes given). Web pages were read on 2026-10-18.

Labels used below:
- **[V]**: verified in the cited source.
- **[D]**: derived by me from verified inputs. The method is stated, so it can be reproduced in the notebook.
- **[NV]**: could not be verified.

---

## 1. GBTDS definition as currently adopted

### 1.1 Governing documents
- **ROTAC Final Report.** "Roman Observations Time Allocation Committee: Final Report and Recommendations", Zasowski & Jha (co-chairs) et al. arXiv:2505.10574 (v1 13 May 2025; v2 23 Oct 2025), report dated 24 April 2025. ADS: `2025arXiv250510574O`. [V]
  - ROTAC recommends the **overguide** GBTDS design (438 days), with two modifications:
    1. The first high-cadence season is shortened by about 2 days, for Galactic Plane Survey epoch 1.
    2. The scheduling of the low-cadence seasons is to be studied further (ROTAC §3.3).
  - The GBTDS Definition Committee report is reproduced as Appendix C.3 of the ROTAC report.
- **STScI RDox GBTDS page** (current implementation): https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey [V]. It says some minor details are "not yet finalized".
- **IPAC page**: https://roman.ipac.caltech.edu/page/gbtds [V].
- **GSFC page**: https://roman.gsfc.nasa.gov/science/Galactic_Bulge_Time_Domain_Survey.html [V].
- **NASA science page**: https://science.nasa.gov/mission/roman-space-telescope/galactic-bulge-time-domain-survey-technical/ [V].

### 1.2 Fields
**Layout.** There are 6 WFI pointings: 5 contiguous bulge fields plus 1 Galactic Center field. Each pointing covers about 0.28 deg², for a total of about 1.7 deg² (6 × 0.281 = 1.69). [V: ROTAC p.5; RDox]

**Field centres.** These are the overguide coordinates from ROTAC Appendix C.3, Table 9, also reproduced on the IPAC page. [V] The RA/Dec and ecliptic columns are my conversion with astropy (ICRS; ecliptic = BarycentricMeanEcliptic J2000). [D]

| Field | l (deg) | b (deg) | RA (deg) | Dec (deg) | RA/Dec (sexagesimal) | ecl. lon λ | ecl. lat β |
|---|---|---|---|---|---|---|---|
| 1 | −0.417948 | −1.200 | 267.3354 | −29.9149 | 17h49m21s −29°54′54″ | 267.68 | **−6.50** |
| 2 | −0.008974 | −1.200 | 267.5772 | −29.5640 | 17h50m19s −29°33′50″ | 267.88 | **−6.14** |
| 3 | 0.400000 | −1.200 | 267.8173 | −29.2127 | 17h51m16s −29°12′46″ | 268.09 | **−5.79** |
| 4 | 0.808974 | −1.200 | 268.0558 | −28.8610 | 17h52m13s −28°51′40″ | 268.29 | **−5.43** |
| 5 | 1.217948 | −1.200 | 268.2928 | −28.5089 | 17h53m10s −28°30′32″ | 268.49 | **−5.08** |
| 6 (Gal. Center) | 0.000000 | −0.125 | 266.5270 | −29.0013 | 17h46m06s −29°00′05″ | 266.95 | **−5.60** |
| (l=0, b=0 reference) | 0 | 0 | 266.4050 | −28.9362 | 17h45m37s −28°56′10″ | 266.84 | −5.54 |

**Inconsistencies to note:**
- The ROTAC prose (§3.3) puts the contiguous region at "ℓ ≈ −0.5°, b ≈ −1.2°". The Table 9 coordinates, however, centre the five fields at l = +0.40°.
- RDox says the mosaic mid-line is at (l, b) = (0.5, −1.4) in spring and (0.35, −1.4) in autumn, and that "some target details" are not final.
- RDox mosaic position angles are about 90.6° in spring and about 270.6° in autumn, so the footprint flips 180° between seasons. RDox also mentions 36 custom mosaics per high-cadence season for small offsets.
- **Advice:** treat the exact centres as provisional to about 0.1–0.2°. This does not matter at the level of TNO orbit-uncertainty work.

### 1.3 WFI field of view and detectors
Source: RDox WFI Quick Reference (https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-quick-reference) and the GSFC WFI technical page (https://roman.gsfc.nasa.gov/science/WFI_technical.html). [V]
- **Detectors:** 18 Teledyne H4RG-10 detectors in a 3 × 6 array.
  - Each is 4096 × 4096 pixels, with 4088 × 4088 active.
  - Pixels are 10 µm, about 300 Mpix in total.
- **Plate scale:** 0.11″/pixel. Each detector covers about 450″ × 450″ on the sky.
- **Field of view:** 0.281 deg² active, 0.320 deg² total focal-plane area including gaps. GSFC gives the footprint as about 0.8° × 0.4°.
- **Field of regard:**
  - Pitch is 54°–126° from the Sun line, with 360° of yaw and ±15° roll.
  - About 59% of the sky is accessible at any one time.
  - These are also given on https://roman.gsfc.nasa.gov/science/observatory_technical.html. [V]
- **Slews:** a 0.4° slew ("short FoV") takes 39.3 s, a 0.8° slew takes 52.5 s, and a 2° slew takes 77.8 s (GSFC observatory technical page). [V]

### 1.4 Seasons: number, length, timing
- **Number:** 10 bulge seasons in the 5-year prime mission. [V: ROTAC App. C.3 p.~3; RDox]
  - 6 are high-cadence: the first three and the last three.
  - 4 are low-cadence: the middle seasons.
- **Length:** the maximum season length is **72 days**, "set by spacecraft pointing limits". [V: ROTAC C.3]
  - Each high-cadence season is allocated **70.5 days**. [V]
  - The first season is reduced by about 2 days for the Galactic Plane Survey. [V]
- **Timing:** "two 72-day seasons centered approximately on the vernal and autumnal equinoxes" (ROTAC C.3 §3.3.5). [V] RDox gives spring as roughly Feb–Apr and autumn as roughly Aug–Oct. [V]
- **Why these dates (my check):** I computed when the Sun–target angle for field 3 lies in the 54°–126° pitch window (astropy, geocentric Sun, 2027). [D]
  - Visible 12 Feb → 26 Apr 2027 (~73 d) and 16 Aug → 29 Oct 2027 (~74 d).
  - The bulge is at λ ≈ 268°, so solar conjunction falls about 21 Dec and opposition about 20 Jun. Elongation is 90° near the equinoxes.
  - The season is therefore observed in quadrature. The 72-day length is simply the 54°–126° pitch band.
  - Roman's ~1.5e6 km offset from Earth changes this by well under 1°.
- **Provisional calendar** (RDox "Roman Observations in the First Two Years of Science Operations": https://roman-docs.stsci.edu/roman-community-defined-surveys/roman-observations-in-the-first-two-years-of-science-operations). [V] The page assumes an 30 Aug 2026 launch and 90-day commissioning, and is marked "Subject to Change".

  | Season | Dates | Type |
  |---|---|---|
  | 1 | 13 Feb – 25 Apr 2027 | high-cadence (2 days reserved for GPS) |
  | 2 | 14 Aug – 30 Oct 2027 | high-cadence |
  | 3 | 13 Feb – 25 Apr 2028 | high-cadence |
  | 4 | 14 Aug – 30 Oct 2028 | low-cadence |

  A search-engine snippet of an older version of this page quoted a 28 Sep 2026 launch assumption and 11 Feb – 20 Apr 2027. The page I fetched has the dates above. [V for current page]

### 1.5 Cadence, exposures, filters

**High-cadence seasons** [V: ROTAC Table 8; RDox; IPAC]
- **F146 (wide, 0.927–2.000 µm):** 12.1-minute cadence per field.
  - Exposure 66.9 s (ROTAC/IPAC; "67 s" in the ROTAC summary).
  - MultiAccum table IM_66_6_V2: effective exposure 66.41 s, integration 69.57 s, 5 resultants, read time 3.16247 s (RDox Imaging MA tables page).
  - About 8390 epochs per field per season (RDox).
- **F087 and F213:** "every 12 hours or faster" (ROTAC). RDox's current table gives **every 6 hours**, 66 s, about 282 epochs per season.
- **Galactic Center field:** observed at the same 12.1-minute cadence.

**Low-cadence seasons** — the sources disagree [V]
- ROTAC, IPAC table and GSFC page: **3-day** cadence in F146 + F213, 4 exposures per visit.
- GSFC intro text: "5-days or faster".
- RDox (current table): **5-day** cadence in F146/F087/F213, about 14 epochs per season, "each six-tile F146 cycle repeated 5 times".
- ROTAC asked for these to be scheduled flexibly, so assume 3–5 days, possibly aperiodic.

**Snapshots** [V: ROTAC; RDox]
- Taken 3 times per season (start, middle, end; about 35 days apart), 30 in total.
- Multiband photometry in F062, F106, F129, F158 and F184, plus grism.
- RDox exposures:
  - F062: 1138 s
  - F106: 272 s
  - F129: 170 s
  - F158: 132 s
  - F184: 132 s
  - grism: 1491 s

**Exposure design** [V: RDox]
- S/N ≈ 100 per exposure in F146 for a source at **F146 = 21.2 AB**.
- The old 2017 WFIRST requirement was 1 mas per measurement at H_AB = 21.6 (see §3).

### 1.6 Total allocation and span
- **Total:** 438 days including overheads (ROTAC/RDox). IPAC says "440 days"; NASA says "~15 months". About 97% of this goes to high-cadence monitoring. [V]
- **Span:** the first season is Feb 2027, if the schedule holds. The last high-cadence seasons fall in years 4–5 of the prime mission. [D, from the 10-season structure]
  - The span is therefore about 4.5 yr (Feb 2027 → about Oct 2031).
  - The exact last-season dates are **[NV]**; no published schedule goes beyond 2028.

---

## 2. Roman orbit and launch status

### 2.1 Launch
- **Launched 30 Aug 2026, 07:26 EDT (11:26:04 UTC)**, on a SpaceX Falcon Heavy from LC-39A at KSC. [V]
  - NASA: https://science.nasa.gov/mission/roman-space-telescope/roman-launch/
  - JPL Horizons header (revised 7 Oct 2026).
- The spacecraft separated from the second stage at 07:57 EDT. [V, NASA page]
- The transfer to L2 takes about 3 months, so L2 arrival is around late Nov/Dec 2026 (NASA commissioning page: https://science.nasa.gov/missions/roman-space-telescope/roman-commissioning/). [V for "three-month journey"] An exact L2 insertion date is **[NV]**.
- Commissioning lasts about 90 days (RDox schedule assumption). [V as an assumption]

### 2.2 Horizons ID and public ephemeris
- **JPL Horizons target `-211`, "Nancy Grace Roman Space Telescope".** [V, queried via the Horizons API on 2026-10-18]
- The ephemeris begins 2026-Aug-30 11:58 UT.
- The trajectory comes from the GSFC Flight Dynamics Facility, delivered 2026-Oct-05. The last tracking data are from 4 Oct; everything after that is prediction.
- The latest file is `RST_EPH_PRED_LONG_2026278_2028278_01`, covering 2026-Oct-05 → 2028-Oct-04.
  - **This is a predicted trajectory, not a reconstruction.** Expect updates after the L2 orbit insertion manoeuvres.
- Example API call:
  `https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='-211'&EPHEM_TYPE=VECTORS&CENTER='500@399'&START_TIME='2027-02-13'&STOP_TIME='2027-04-25'&STEP_SIZE='1 d'`
- Usable observatory codes for an MPC-style or `astroquery` workflow: geocentric `500@399`, or spacecraft `@-211`. Whether an MPC observatory code has been assigned for Roman is **[NV]**.
- The GSFC observatory page links pre-launch orbit files for 4 cases, which assumed a nominal launch of 30 Oct 2026; those files are superseded. [V]

### 2.3 Halo-orbit characteristics
- **Published descriptions:**
  - Sun–Earth L2 quasi-halo orbit at about 1.5e6 km from Earth (NTRS 20210018417).
  - "Particularly small amplitude halo orbits, compared to past missions like JWST". This is driven by daily White Sands downlink geometry (≥1.3 TB/day) (Konitzer et al., AAS preprint, NTRS 20240004229).
  - No official amplitude or period number was found in NASA/STScI documents: **[NV]** as an official figure.
- **My numbers from the Horizons `-211` prediction**, Dec 2026 – Oct 2028, daily. [D] Method: geocentric ecliptic J2000 vectors of −211 and the Sun, rotated into an Earth-centred frame with x pointing anti-sunward.
  - **Earth–Roman distance:** 1.20–1.73 × 10⁶ km (mean 1.55 × 10⁶ km).
  - **Anti-sunward offset x:** 1.19–1.72 × 10⁶ km.
  - **In-ecliptic transverse excursion y:** about ±0.78 × 10⁶ km.
  - **Out-of-ecliptic excursion z:** about −0.22 to +0.17 × 10⁶ km.
  - **Sun–Earth–Roman angle:** 7°–30°.
  - **Libration period:** about 180 days (zero crossings of y about every 90 days).
  - Caveat: this is the pre-insertion predicted trajectory.
- **Parallax lever arm for TNOs** [D]: baseline B ≈ 1.2–1.7 × 10⁶ km ≈ 0.008–0.012 au.
  - For a TNO at 40 au, the Earth–Roman parallax is B/Δ ≈ 2.5 × 10⁻⁴ rad ≈ **50″**. At 100 au it is about 20″.
  - This agrees with Giovinazzi et al. (2021): "~10″–100″".

---

## 3. Astrometric precision, PSF, saturation and depth

### 3.1 PSF and depth
Source: GSFC WFI technical page; RDox Quick Reference. Values are tied to Roman Technical Information Repository v1.2 (July 2025). [V]
- **F146 PSF FWHM:** 0.105″, i.e. about 1.0 pixel, so the PSF is undersampled. Other filters range from 0.058″ (F062) to 0.169″ (F213).
- **5σ point-source depth, F146:**
  - 25.37 AB in a "57 s" exposure (the text says "55 s").
  - 28.01 AB in 1 hr.
  - Both assume 2× minimum zodiacal light.
  - Scaling to 66.4 s gives about 25.45 AB. [D, t^½ scaling, rough] The bulge background is higher than the zodiacal-only assumption, so the real single-exposure limit in the GBTDS will be shallower.
- **Exposure-design anchor:** S/N = 100 at F146 = 21.2 AB per 66 s exposure (RDox). [V]

### 3.2 Saturation
Source: GSFC table "time to saturation for point imaging", run 2021-06-14: https://roman.gsfc.nasa.gov/science/apttables2021/performance_tables/2021/run_time_to_saturation_20210614.txt [V]

| F146 point source | Time to saturate the peak pixel |
|---|---|
| 15.0 AB | 2.5 s |
| 16.0 AB | 6.4 s |
| 17.0 AB | 16.0 s |
| 18.0 AB | 40.1 s |

Interpolated thresholds [D]:
- **≈18.5 AB** saturates within the full 66.4 s.
- **≈16.0 AB** saturates within the first two reads (6.3 s). Sources fainter than this still give a usable ramp slope.
- **≈15.2 AB** saturates within the first read.

McKinnon & van der Marel (2026, PASP 138, 044507; ADS `2026PASP..138d4507M`) put hard saturation at "≲15.5 mag for most Roman filters and exposure times". [V] TVAC bright-star tests are in Louie et al. 2026, arXiv:2607.18419 (`2026arXiv260718419L`). [V]

### 3.3 Astrometric precision
**WFIRST Astrometry Working Group, Sanderson, R. E., Bellini, A., Casertano, S., et al. 2019**, "Astrometry with the Wide-Field Infrared Space Telescope", *JATIS* 5(4), 044005. doi:10.1117/1.JATIS.5.4.044005, arXiv:1712.05420, ADS `2019JATIS...5d4005W`. [V] Its Table 1 numbers, described in the paper as "order-of-magnitude only":
- Single exposure, well-exposed point source: **0.01 pixel ≈ 1.1 mas**.
- 100 exposures: 0.1 mas.
- Absolute astrometry (with Gaia): 0.1 mas.
- Microlensing survey, per image: **1 mas**.
- Microlensing survey, whole survey: 3–10 µas systematic.
- It also quotes the 2017 microlensing requirements:
  - EML 19: "statistical precision of 1 mas per measurement for H_AB = 21.6".
  - EML 20: 10 µas systematic over the survey.
  - The jitter requirement was ≤14 mas rms. The current observatory page gives jitter 8 mas and drift 8 mas. [V]

**Precision as a function of magnitude (McKinnon & van der Marel 2026)** [V]
- σ_pos = k / SNR_flux, with k(F146) = 0.792 px ≈ **87 mas / SNR**.
- A 1% pixel floor (1.1 mas) is added in quadrature.
- Fainter than about 24 AB (for a 193 s exposure), background dominates and σ_pos rises faster.
- They warn that chromatic PSF variation across the wide F146 band may cause systematics.
- Recipe [D]: σ(F146 = 21.2) ≈ √(0.87² + 1.1²) ≈ 1.4 mas per 66 s exposure. At S/N = 10 it is about 9 mas. At S/N = 5 it is about 17 mas.

**Gould's earlier assumption (2014)** [V]: σ_ast = 2σ_psf/SNR = **106 mas/SNR** per exposure, with SNR = 10^{0.4(26.1 − H_Vega)}, for the older WFIRST design (W149, 52 s exposures, 15-minute cadence). This is consistent to about 20% with the newer k-factor.

**Absolute frame.** Gaia supplies the reference frame. Gould (2014) notes Gaia covers millions of stars in the fields. The 2019 Working Group report says about 150 Gaia stars with 17 < G < 19 per WFIRST field. [V] Gaia DR4/DR5 plus Roman will tie TNO astrometry to the frame at well below 1 mas.

### 3.4 Moving-object caveats (important for the notebook)
- **No non-sidereal tracking.** "Roman is unable to track objects moving at non-sidereal rates" (Masiero 2025). [V] Holler et al. (2018) had argued for 30 mas/s tracking, but it was not implemented.
- **Pipeline rejection of moving flux.** The default ramp-fitting/jump-detection pipeline preferentially rejects flux from moving objects as cosmic rays.
  - Masiero, J. 2025, "Technical Memo: The impact of Nancy Grace Roman Telescope's default image processing on the detectability of moving Solar system objects", arXiv:2509.20469, ADS `2025arXiv250920469M`. [V]
  - KBOs move about 2″/hr, i.e. about 1 pixel in 300 s, so they are only mildly affected. NEOs are strongly affected.
  - The same point is made in R. Pike (MPC) slides, Sagan Summer Workshop, 23 Jul 2026: https://vmnexsci.ipac.caltech.edu/workshop/2026/talks/20260723/Invited_talk_slides_RosemaryPike_131.pdf [V]
- **TNO rate during GBTDS seasons** [D]: the bulge is observed near quadrature, where Earth's velocity is nearly along the line of sight.
  - The TNO's own ~4.7 km/s at 40 au gives about 0.6″/hr at mid-season.
  - At the season edges (±36° from quadrature), Earth's transverse component is about 30 sin 36° ≈ 18 km/s, giving about 2–2.5″/hr.
  - Motion within one 66 s exposure is therefore ≲0.05″ (≲0.4 pixel), so effectively no trailing.
  - Gould (2014) adopts an rms reflex velocity of about 10 km/s and estimates about 1° of motion per 72-day season. [V] About 60% of KBOs stay in the field for the whole season (his Eq. 4, for the 10-field old layout).
  - With the current 5-field layout (about 0.8° × 2° contiguous), a much smaller fraction will stay in the field all season. **[D: not computed; should be simulated in the notebook]**

---

## 4. Literature (all checked on ADS)

### 4.1 Roman/WFIRST Solar System science
- **Holler, B. J., Milam, S. N., Bauer, J. M., et al. (33 authors) 2018**, "Solar system science with the Wide-Field Infrared Survey Telescope", *JATIS* 4(3), 034003. doi:10.1117/1.JATIS.4.3.034003, arXiv:1709.02763, `2018JATIS...4c4003H`. [V]
  - §6.2 covers KBOs in the microlensing survey: detections down to V ≈ 30.2 (about 11 km at 40 au, albedo p = 0.04), and satellites down to V ≈ 31.0.
  - §5.5 covers targeted occultations from L2: a guide-window readout of about 5.8 Hz and a 10 Hz Fresnel-scale requirement for KBOs. "KBOs as small as ~15 km can be resolved … although … considerable positional uncertainties in the predictions."
  - It uses the pre-2020 design (7–8 fields, W149, 52 s).
- **Holler, B. J., Cosentino, R. G., Schultz, W. C., Brandt, T. D., Masiero, J. R., Sharkey, B. N. L. 2025**, "The Roman Space Telescope as a Planetary Defense Asset", *PASP* 137, 105004. doi:10.1088/1538-3873/ae0fe5, arXiv:2508.14412, `2025PASP..137j5004H`. [V]
  - Claims Roman can improve NEO orbits by 2–3 orders of magnitude.
  - Its abstract still says "slated to launch in 2026 October", which is now superseded.
- **Masiero 2025**, arXiv:2509.20469 (above). [V]
- **Benecchi, S. D., Porter, S. B., Verbiscer, A. J., et al. 2025**, "Characterization of the Deep, Extended Kuiper Belt in the Galactic Disk", arXiv:2503.02909, `2025arXiv250302909B`. [V]
  - A Roman survey white paper aimed at the New Horizons search region, with up to about 900 KBOs. It is not part of the GBTDS.
- **Roman Cycle 1 approved program 19019**, "An Archival Deep Drilling Kuiper Belt Search in the Galactic Bulge Time Domain Survey", PI Kevin Napier (CfA). https://roman.ipac.caltech.edu/cycle1-approved-programs/19019 [V]
  - Aims for about 1000 KBOs to F146 = 28.0 in the first three seasons, via shift-and-stack.
  - Says the field sits at "ecliptic latitude of 5–6 degrees".
  - Pike's slides list a second Cycle 1 small-body program: "Recovering NEO Streaks from Roman WFI Data for Planetary Defense", PI Arturo Sasli (Program ID 19035 per the slides). [V as stated on slides]
- **ROTAC GBTDS white papers** (Appendix C.3 table, no refereed version found) [V as listed]:
  - SP8: "Rotational Light Curves and Phase Curves of Small Solar System Bodies", T. Terai.
  - SP35: "Surveying the rotational properties of TNOs", L. Molnár.

### 4.2 Gould KBO papers (title correction)
- **Gould, A. 2014**, "**WFIRST Ultra-Precise Astrometry I: Kuiper Belt Objects**", *Journal of the Korean Astronomical Society* 47(6), 279. doi:10.5303/JKAS.2014.47.6.279, arXiv:1403.4241, `2014JKAS...47..279G`. [V]
  - "Paper II" is about asteroseismology, not asteroids: Gould, Huber, Penny & Stello 2015, JKAS 48, 93, `2015JKAS...48...93G`.
  - Results:
    - KBO detection to H_Vega = 28.2 over an effective area of about 17 deg².
    - Fractional period errors of about 1.5% × 10^{0.4(H−28.2)}, for about 5000 KBOs.
    - Binary companions detectable to H_Vega ≈ 29 (D ≈ 7 km).
    - More than 1000 serendipitous stellar occultations by KBOs.
  - Assumptions: 10 × 0.28 deg² fields, six 72-day campaigns centred at quadrature, 15-minute cadence, 0.11″ pixels, and σ_ast = 106 mas/SNR. The current design is quite different: 5 + 1 fields at a 12.1-minute cadence.
  - He also says current ground microlensing surveys "can make a significant start" on KBOs.
- **Gould, A. 2016**, "Microlensing by Kuiper, Oort, and Free-Floating Planets", *JKAS* 49, 123, `2016JKAS...49..123G`. [V]
- **Gould, A., Zang, W., Mao, S., Dong, S. 2021**, "Masses for free-floating planets and dwarf planets", *RAA* 21, 133, `2021RAA....21..133G`. [V] Argues for an L2–Earth baseline of about 0.01 au for microlens parallax.

### 4.3 L2–Earth parallax and space-based astrometry for TNO orbits and occultations
- **Giovinazzi, M. R., Blake, C. H., Bernardinelli, P. H. 2021**, "Enhancing Ground-based Observations of Trans-Neptunian Objects Using a Single-epoch Parallax Measurement from L2", *PASP* 133, 114401. doi:10.1088/1538-3873/ac2e0e, arXiv:2110.13327, `2021PASP..133k4401G`. [V]
  - **This is the key paper for the notebook.**
  - The ~1.5e6 km baseline gives parallax of about 10–100″.
  - Three ground epochs over 24 hr plus one L2 epoch give Δa/a of about 10⁻², with σ_distance about 10⁻² au.
  - Assumes σ ≈ 25 mas for the relative Subaru-to-L2 offset (σ_Subaru = 25 mas; JWST <5 mas).
  - Recommends that ground and L2 epochs be taken within about a week of each other.
- **Porter, S. B., Spencer, J. R., Verbiscer, A., et al. 2022**, "Orbits and Occultation Opportunities of 15 TNOs Observed by New Horizons", *PSJ* 3, 23. doi:10.3847/PSJ/ac3491, `2022PSJ.....3...23P`. [V]
  - A worked example of high-parallax spacecraft astrometry shrinking TNO orbit uncertainties for occultation predictions.
- **Desmars, J., Camargo, J. I. B., Braga-Ribas, F., et al. 2015**, "Orbit determination of trans-Neptunian objects and Centaurs for the prediction of stellar occultations", *A&A* 584, A96. doi:10.1051/0004-6361/201526498, `2015A&A...584A..96D`. [V]
  - The standard reference for occultation-oriented orbit fitting (the NIMA approach).
- **Sicardy, B., Braga-Ribas, F., Buie, M. W., Ortiz, J. L., Roques, F. 2024**, "Stellar occultations by trans-Neptunian objects", *A&ARv* 32, 6. doi:10.1007/s00159-024-00156-x, `2024A&ARv..32....6S`. [V] A review, including prediction accuracy requirements.
- **No paper found** that specifically treats Roman-GBTDS-derived TNO orbits for occultation predictions, or Roman-versus-ground uncertainty propagation. I searched ADS for occultation + Roman/WFIRST + TNO/Kuiper. The notebook would be novel on this point. [NV, absence]

### 4.4 Small bodies in microlensing and bulge surveys
- **MOA:** Cordwell, A. J., Rattenbury, N. J., Bannister, M. T., et al. 2022, "Asteroid Lightcurves from the MOA-II Survey: a pilot study", *MNRAS* 514, 3098, `2022MNRAS.514.3098C`. [V]
- **MOA:** Cowan, P., Bond, I. A., Reyes, N. H. 2023, "Towards asteroid detection in microlensing surveys with deep learning", *Astronomy and Computing* 42, 100693, `2023A&C....4200693C`. [V] See also Cowan et al. 2025, EPSC-DPS abstract `2025epsc.conf.1073C`.
- **KMTNet:** Huang, H., Yang, H., Lee, C.-U., et al. 2026, "Measuring Asteroid Rotation Periods Using the KMTNet Bulge Survey Data", *PASP* 138, 074401, `2026PASP..138g4401H`. [V]
- **K2 Campaign 9** (bulge microlensing campaign): Podlewska-Gaca, E., Poleski, R., Bartczak, P., McDonald, I., Pál, A. 2021, "Determination of Rotation Periods for a Large Sample of Asteroids from the K2 Campaign 9", *ApJS* 255, 4, `2021ApJS..255....4P`. [V] It covers 1026 asteroids, with periods for 188.
- **No refereed paper found** reporting TNO/KBO detections in OGLE, MOA, KMTNet or K2 C9 bulge data. [NV, absence] Gould (2014) suggested it was feasible, but I found no published result.

### 4.5 Microlensing-survey context (not Solar System)
- **Penny, M. T., Gaudi, B. S., Kerins, E., et al. 2019**, "Predictions of the WFIRST Microlensing Survey. I. Bound Planet Detection Rates", *ApJS* 241, 3, `2019ApJS..241....3P`. [V]
- **Roman Galactic Plane Survey Definition Committee 2025**, arXiv:2511.07494, `2025arXiv251107494G`. [V]

---

## 5. Ecliptic latitude of the GBTDS fields
- **β ranges from −5.1° to −6.5° across the 6 fields** (§1.2 table), with ecliptic longitude λ ≈ 267–268.5°. [D, astropy J2000 mean ecliptic]
  - Field 6 (Galactic Center) is at β = −5.60°.
  - The contiguous strip runs from −6.50° (field 1, lowest l) to −5.08° (field 5).
- **Sgr A\* / (l, b) = (0, 0):** β = −5.54°. This confirms the "~ −5.6°" figure.
- **Cross-checks** [V]:
  - The Napier Cycle 1 abstract says "ecliptic latitude of 5–6 degrees".
  - Pike's slides say "approximately 5–7 degrees off the ecliptic plane".
  - Gould (2014) says "a few degrees from the ecliptic".
- **Implication** [D]: the fields sit about 5–6.5° south of the ecliptic, inside the TNO "hot" population but above the core of the cold classical belt (inclination ≲ 5°).
  - KBO sky density there is lower than on the ecliptic. Pike's slides note community interest in adding an ecliptic GAS pointing for that reason.

---

## 6. Summary of main uncertainties
1. **Exact field centres and mosaic offsets.** There are differences at the ~0.1° level between ROTAC Table 9, the ROTAC prose and RDox, and RDox says details are not final.
2. **Low-cadence season cadence.** Sources give 3 days or 5 days; the scheduling is unsettled.
3. **F087/F213 cadence.** ROTAC says 12 hr or faster; RDox's current table says 6 hr.
4. **Schedule after 2028.** The 4.5-yr span is inferred; the start in Feb 2027 depends on commissioning.
5. **Roman's trajectory.** The Horizons −211 orbit is a prediction made before L2 insertion. Halo amplitude and period are my derivations; no official figure was found.
6. **Astrometric precision for faint moving sources in crowded bulge fields.** Existing numbers come from requirements and simulations: 1 mas at H_AB ≈ 21–21.6, k/SNR plus a 1.1 mas floor. Real performance for faint TNOs in crowded, high-background bulge fields is not established.
7. **Pipeline and data products.** The default processing may reject or flag moving-object flux (mild for TNOs). No non-sidereal tracking.
8. **No published work** on Roman-derived TNO orbits for occultation prediction, and no TNO detections from ground bulge surveys.
