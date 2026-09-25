# Clinical Gaps in Scapula Morphology Addressable by a Bilateral SSM

## 1. Glenoid Version: Normal-Population Variation and Contralateral Reference Validity

### Takeaway
Glenoid version varies widely in healthy individuals (ranges spanning ≥20°), 2D measurements systematically underestimate values compared to 3D, and while the contralateral scapula is widely used as a surgical template, the bilateral relationship has not been validated in full 3D shape space.

### Cited Findings
- Mean glenoid version in a normal Indian population was 0.07 ± 5.38°, ranging from −11° to +10.85° — [Three Dimensional Anthropometric Analysis of Glenoid Anatomy in Normal Indian Population (Springer, 2021)](https://link.springer.com/article/10.1007/s43465-020-00321-1)
- A second Indian population study found a mean glenoid version of 0.05 ± 9.05°, indicating even greater individual variability — [3D CT-based study of glenoid morphology in Indian population (ScienceDirect, 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0976566220300825)
- A regression study found males would be expected to exhibit 8.4° more glenoid retroversion than females — [Glenoid version and size: does gender, ethnicity, or body size play a role? (PubMed, 2016)](https://pubmed.ncbi.nlm.nih.gov/27106214/)
- 3D CT measurement of glenoid version is accurate within 1.0° ± 0.7° vs. actual specimens; 2D images may misrepresent 3D anatomy — [The Normal 3D Gleno-humeral Relationship and Anatomy of the Glenoid Planes (JBSR, 2018)](https://jbsr.be/articles/10.5334/jbsr.1346)
- Mean preoperative glenoid retroversion on 2D CT was 11.9° ± 9.6° vs. 15.1° ± 10.6° on 3D planning software — a systematic 3.2° underestimate — [Variability and reliability of 2D vs. 3D glenoid version measurements with 3D planning software (ScienceDirect, 2021)](https://www.sciencedirect.com/science/article/abs/pii/S1058274621006030)
- The 2D-3D discrepancy led to an adjustment in implant choice in 7 out of a reported cohort of patients — same source
- Paired right and left scapulae are **not statistically symmetrical** regarding mean glenoid version, inclination, and width, despite strong inter-side correlation coefficients — [Premorbid glenoid anatomy reconstruction from contralateral shoulder 3D measurements: analysis of 260 shoulders (PubMed, 2023)](https://pubmed.ncbi.nlm.nih.gov/37852431/)
- Mean bilateral differences across 260 scapulae: 2 mm in offset, 2° in inclination, 2° in version; maximum bilateral differences were 6 mm, 6°, and 8° respectively — same source
- In 96% of scapula pairs the inclination difference and in 94% the version difference were < 5°, confirming practical utility for arthroplasty planning when the contralateral side is healthy — same source
- A routine neutral glenoid version target in reverse shoulder arthroplasty is under challenge, with advocates calling for patient-specific, fixation-oriented approaches — [Routine neutral glenoid version targets in RSA: Time for a patient-specific approach (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13502450/)

### Inferences
- Because the contralateral scapula shows maximum bilateral differences of up to 8° in version, using it as a universal surgical template may introduce clinically meaningful error in outlier patients who cannot be identified pre-operatively from 2D measurements alone.
- A paired bilateral SSM could quantify where in shape space bilateral version differences are largest, providing a data-driven boundary for when the contralateral reference is and is not reliable.
- The existing literature uses discrete measurements; no study has assessed bilateral symmetry in continuous 3D shape space using an SSM.

### Gaps
- No published study has characterized bilateral scapula symmetry in full 3D shape space (i.e., using SSM distance metrics or residual shape after mirroring), as distinct from individual landmark measurements.
- The 260-shoulder study by Bouliane et al. (2023) predates SSM-based shape analysis; its findings on statistical asymmetry are limited to 2D-derived scalar measurements.
- The degree to which outlier bilateral asymmetries cluster in specific shape-model modes (e.g., mode 1 size, mode 3 acromial shape) has not been investigated.

---

## 2. Bigliani Acromion Classification: Limitations and SSM-Based Replacement

### Takeaway
The Bigliani classification has been shown to have poor inter-observer reliability, is highly sensitive to which imaging slice is used, and correlates with age — suggesting it may reflect remodelling rather than innate morphology. No data-driven SSM-based replacement specific to the acromion has been published.

### Cited Findings
- The Bigliani classification showed only slight inter-observer reliability: kappa ≈ 0.25 — [Reliability of Bigliani's Classification using MRI (Morthoj, 2022)](https://www.morthoj.org/2022/v16n3/bigliani-s-classification-acromial.pdf)
- Hooked acromion (Type III) prevalence varied between 7.1% (slice S-1, lateral edge), 16.1% (S-2), and 37.5% (S-3, near AC joint) depending on which slice was evaluated — demonstrating extreme slice-position dependency — [A reliable method for classifying acromial shape (Taylor & Francis, 2015)](https://www.tandfonline.com/doi/full/10.1080/23335432.2015.1014847)
- A 2025 study of 420 patients found a significant association between acromial morphology and age, suggesting acromial shape evolves over life rather than being purely innate — [Acromial morphology: reliability of CT-based assessment and association with age (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S2666638326001301)
- Even though the Bigliani classification is widely used, there is significant disagreement in the literature regarding incidences of morphological types, linked to lack of standardised objective criteria — same source
- Good intrarater but poor inter-rater reliability was confirmed for Bigliani classification using standardised 3D CT reconstructions — same source
- Acromial slope has been shown to vary by up to 16° in patients with subacromial impingement syndrome, yet the Bigliani system collapses this continuous variation into 3 ordinal types — [Three-dimensional scapular morphology is associated with rotator cuff tears (PMC via search summary, 2021)](https://www.jshoulderelbow.org/article/S1058-2746(24)00865-6/fulltext)
- The first five shape modes of a 3D scapular SSM included acromial shape and slope (4% of variance) and acromial overhang (2% of variance) as distinct independent variation modes — [Morphologic variations of the scapula in 3-dimensions: a statistical shape model approach (JSES/PubMed, 2018)](https://pubmed.ncbi.nlm.nih.gov/30100175/)

### Inferences
- The continuous modes of acromial shape variation in an SSM (principally modes 3 and 5 in the Gilat 2018 model) could provide a data-driven, observer-independent replacement for the Bigliani ordinal types.
- Age-related remodelling of the acromion implies that cross-sectional morphological databases will conflate true shape variation with ageing artefact; a prospective or age-stratified approach is needed to disentangle these.

### Gaps
- No published study has used an SSM to derive a data-driven, continuous or cluster-based acromion classification that replaces or supplements Bigliani types — this is an explicit research gap.
- It is unknown whether the SSM modes that capture acromial shape relate to the clinically meaningful dimension of sub-acromial clearance better than Bigliani types.

---

## 3. Bilateral Scapula Asymmetry in 3D Shape Space

### Takeaway
Existing work on bilateral scapula asymmetry uses either kinematic motion data or discrete landmark measurements; no study has quantified asymmetry in continuous 3D shape space using a paired-specimen SSM.

### Cited Findings
- A 3D motion analysis study calculated a symmetry angle throughout shoulder elevation to quantify kinematic scapular asymmetry in participants with and without shoulder impingement syndrome — [Scapular asymmetry in participants with/without shoulder impingement syndrome; 3D motion analysis (ScienceDirect, 2016)](https://www.sciencedirect.com/science/article/abs/pii/S0268003316301292)
- Healthy overhead athletes showed dominant-side scapulae more internally rotated and anteriorly tilted than the nondominant side — [Asymmetric resting scapular posture in healthy overhead athletes (PubMed, 2009)](https://pubmed.ncbi.nlm.nih.gov/19030133/)
- Minor scapular asymmetries exist in healthy individuals; clinicians should be cautious about treating them in the absence of symptoms — [Minor Scapular Asymmetries May be Normal (Sports Medicine Research commentary, 2009)](https://www.sportsmedres.org/minor-scapular-asymmetries-may-be-norma/)
- One scapula study explicitly noted that "scapulae analyzed were unpaired, which precluded side-to-side comparison and limited the investigation of bilateral anatomical asymmetry" — [Bilateral scapula shape asymmetry gap, noted in search summary from ScienceDirect comprehensive morphology study]
- A 369-scapula CT study found relatively low variability in shape among healthy scapulae but did not include paired bilateral comparison — [Three-Dimensional Geometry of the Normal Scapula: A Software Analysis (PubMed, 2025)](https://pubmed.ncbi.nlm.nih.gov/41875222/)

### Inferences
- A paired bilateral dataset of 22 specimens (44 scapulae) could be the first to characterise bilateral asymmetry in terms of SSM shape coordinates, identifying which shape modes show the most left-right divergence and whether this correlates with dominant/non-dominant side or sex.
- Because existing studies explicitly cite unpairing as a limiting factor, a paired dataset is a directly addressable gap.

### Gaps
- No study in the available literature has measured bilateral asymmetry of scapulae in 3D shape space (i.e., SSM shape coordinates or surface distance after mirroring registration), only in kinematics or discrete measurements.
- The relationship between static bony asymmetry (morphology) and dynamic kinematic asymmetry has not been characterised.

---

## 4. Scapula Morphology, Shoulder Impingement, Rotator Cuff Tears, and Arthroplasty Outcomes

### Takeaway
3D SSM studies have shown that cranial glenoid orientation and subacromial narrowing are associated with rotator cuff tears, but the 2024 literature challenges a link between global scapular shape and full-thickness tears — indicating the relationships are morphologically specific and incompletely characterised.

### Cited Findings
- The first five scapular SSM modes captured: overall size (72%), rotation of the coracoacromial complex (5%), acromial shape and slope (4%), scapular spine shape (2%), acromial overhang (2%) — [Morphologic variations of the scapula in 3D: SSM approach (JSES 2018)](https://pubmed.ncbi.nlm.nih.gov/30100175/)
- Cranial orientation of the glenoid and subacromial narrowing are strongly associated with rotator cuff tears in 3D analysis — [Three-dimensional scapular morphology is associated with rotator cuff tears and alters the supraspinatus moment arm (PMC, 2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8161464/)
- A 2024 JSES study found that scapular morphology is associated with certain patterns of glenohumeral osteoarthritis but **not with full-thickness rotator cuff tears** — contradicting a simple structural link — [Scapular morphology is associated with OA patterns but not with full-thickness rotator cuff tears (JSES 2024)](https://www.jshoulderelbow.org/article/S1058-2746(24)00865-6/fulltext)
- A 2024 Journal of Orthopaedic Research simulation study showed that scapular morphology variation meaningfully affects reverse total shoulder arthroplasty (RSA) biomechanics — [Scapular morphology variation affects RSA biomechanics (J Orthop Res 2024)](https://onlinelibrary.wiley.com/doi/full/10.1002/jor.25801)
- Scapula anatomy (cranial anatomy specifically) has been shown to influence impingement-free range of motion after RSA using databases of up to 10,000 patients — [Scapula anatomy influences simulated impingement-free ROM in RSA (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1058274625004756)
- A machine learning study found that 8 features predicted TSA outcomes with 87.1% accuracy, including glenoid retroversion and humeral head-acromion distance — [ML analysis of TSA outcomes (PubMed, 2024)](https://pubmed.ncbi.nlm.nih.gov/39318416/)
- SSM can match any scapula shape with joint kinematics to measure rotator cuff muscle mechanical advantage, enabling patient-specific preventative therapies — [same SSM approach study 2018]

### Inferences
- The contradiction between the 2021 (positive association) and 2024 (no association) studies on rotator cuff tears suggests that the relationship depends on which morphological features are examined and how "rotator cuff tear" is operationalised (partial vs. full-thickness, specific tendons).
- A new SSM on a well-controlled paired dataset could resolve this by isolating specific modes of variation (e.g., acromion overhang vs. glenoid orientation) and testing them independently against pathology markers.

### Gaps
- No study has examined whether bilateral shape asymmetry (dominant vs. non-dominant) predicts rotator cuff pathology risk asymmetrically.
- The link between shoulder morphology and arthroplasty outcomes has been modelled computationally but not validated against long-term clinical outcome data stratified by SSM shape mode.

---

## 5. Population-Specific (Ethnic/Geographic) Variation in Scapula Morphology via 3D SSM

### Takeaway
Japanese, Chinese, and Vietnamese populations show distinct morphological profiles compared to North American/European populations, but most ethnic groups outside East Asia lack validated 3D SSM characterisation, and no multi-ethnic comparative SSM exists.

### Cited Findings
- The smaller shoulder morphology typical of East Asian (especially Japanese) populations results in challenges when implants designed from North American and European anatomies are used — [Population-specific SSM of the Japanese shoulder: scapula + humerus by sex and stature (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273566/)
- A 2025 Japanese SSM (n = 114 arthroplasty patients) showed PC1 = global scaling, PC2 = coracoid variation, PC3 = scapular aspect ratio; significant sex and stature differences in scapula height (female-170: 132 ± 8 mm; male-170: 165 ± 7 mm) — same source
- Scapula height, width, and glenoid parameters differ significantly between Japanese and Western normative data — same source
- Vietnamese acromion morphology shows distinct profiles compared to other Asian populations — [Morphology and Morphometry of the Acromion Process in Vietnamese Scapulae (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13090134/)
- A 2025 study in a homogeneous (presumed single-country) cohort found significant interpopulation variation in morphological and anthropometric parameters — [Comprehensive analysis of bony scapula morphology and anthropometry (PubMed, 2025)](https://pubmed.ncbi.nlm.nih.gov/40604525/)
- Sex-related differences identified in 3D: men show a longitudinally longer scapular body, posteriorly inclined lateral morphology, and anteriorly oriented acromion with increased anteroposterior diameters — [Scapular morphological variations and sex-related and generational differences using homologous model (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S2666639125000677)

### Inferences
- Published SSMs draw almost exclusively from North American, European, and East Asian cadaveric or clinical registries; South Asian, African, Middle Eastern, and South American populations have no validated 3D SSM.
- Indian 3D CT data exists for glenoid parameters in isolation, but no full-scapula SSM for this large population has been published.
- A paired bilateral dataset from a cohort distinct from the reference databases (e.g., European, South Asian) would add both bilateral and population-specific novelty simultaneously.

### Gaps
- No multi-ethnic comparative 3D SSM of the scapula has been published that allows direct shape-mode comparison across ethnic groups.
- South Asian, African, and Middle Eastern populations are absent from the published SSM literature; 2D morphological studies exist but do not capture full shape variation.
- It is not known whether the principal modes of variation are population-invariant or population-specific (i.e., whether a European SSM describes Indian or African shape space accurately).

---

## 6. Clinical Measurements from 2D vs. 3D — Glenoid Inclination, Acromio-Humeral Distance, Scapular Spine Angle

### Takeaway
Key clinical parameters — glenoid inclination, glenoid version, and acromio-humeral distance — are routinely estimated from 2D imaging but are measurably inaccurate compared to 3D, and clinical decisions have been shown to change when 3D data replaces 2D. An SSM could provide patient-specific 3D references against which 2D measurement error can be quantified.

### Cited Findings
- Mean glenoid retroversion on 2D CT: 11.9° ± 9.6°; on 3D planning software: 15.1° ± 10.6° — a systematic ~3.2° underestimate — [Variability and reliability of 2D vs. 3D glenoid version measurements (ScienceDirect, 2021)](https://www.sciencedirect.com/science/article/abs/pii/S1058274621006030)
- Mean glenoid superior inclination on 2D CT: 10.7° ± 8.6°; on 3D: 8.9° ± 9.9° — an overestimate in the other direction — same source
- A change in implant choice was made in 7 patients when 3D planning replaced 2D in a surgical cohort — same source
- 3D measurement of acromio-humeral distance (AHD) is significantly smaller than 2D measurement due to oblique projection artefact; 3D CT reconstruction provides more accurate results — [Evaluation of 3D AHD in standing position vs. conventional methods (PMC, 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7510276/)
- AHD varies between supine CT and upright standing measurement due to gravitational effects — same source; upright CT scanners can address this but are not standard
- In vivo 3D CT in Chinese populations determined minimum AHD and showed significant age and sex associations that 2D methods would miss — [3D CT minimum AHD in Chinese population (PMC, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9694460/)
- Two-dimensional glenoid inclination measurement has acceptable inter-observer reliability but is less accurate than 3D measurement; the β-angle on reformatted CT is the most accurate 2D method with mean difference of only 1° (SD 0.5°) vs. 3D — [Comparison of glenoid inclination angle using different clinical imaging modalities (ScienceDirect, 2015)](https://www.sciencedirect.com/science/article/abs/pii/S1058274615003754)
- Automated 3D measurement of glenoid version and inclination in arthritic shoulders has been demonstrated as feasible — [Automated 3D Measurement of Glenoid Version and Inclination in Arthritic Shoulders (JBJS, 2017)](https://www.ovid.com/jnls/jbjsjournal/abstract/10.2106/jbjs.16.01122)

### Inferences
- A population-level 3D SSM with bilateral pairs could generate reference distributions for glenoid version, inclination, AHD, and scapular spine angle that are missing from current clinical tools.
- Because the bilateral SSM allows the healthy side to serve as an internal per-patient control, it could quantify how much 2D measurement error is patient-specific versus systematic — a use case not covered by any existing study.
- The acromio-humeral distance is closely tied to acromion morphology (Bigliani type); an SSM that captures both simultaneously would allow integrated clinical measurement beyond what either standalone approach provides.

### Gaps
- No study has used an SSM to generate normative 3D reference ranges for glenoid inclination, glenoid version, acromio-humeral distance, and scapular spine angle simultaneously, with uncertainty bounds.
- Scapular spine angle measurement in 3D has been described but not systematically validated against clinical outcomes in a shape-model framework.
- The acromio-humeral distance as a continuous function of 3D scapula shape mode (i.e., which modes of variation maximally change AHD) has not been modelled.
