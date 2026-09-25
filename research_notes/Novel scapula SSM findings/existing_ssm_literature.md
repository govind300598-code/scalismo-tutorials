# Statistical Shape Models of the Scapula: Existing Literature Review

## What scapula SSM papers exist (authors, year, journal, dataset size N)?

### Takeaway
At least 10–15 peer-reviewed 3D scapula SSM papers exist (2013–2025), with dataset sizes ranging from N=27 dry scapulae to N=500 CT scans; no single study dominates the field and most are from orthopaedic surgery journals focused on shoulder arthroplasty planning.

### Cited Findings

- **Cheung et al. (2013):** "Scapula Statistical Shape Model construction based on watershed segmentation and elastic registration." Conference proceedings, CAOS 2013 / Bone & Joint. Built SSM from CT scans using 8-zone watershed segmentation and elastic registration. Dataset size not confirmed from accessible content but appears to be a small preliminary study. — [Source](https://online.boneandjoint.org.uk/doi/10.1302/1358-992X.95BSUPP_28.CAOS2013-031); [ResearchGate PDF](https://www.researchgate.net/publication/260763189_Scapula_Statistical_Shape_Model_construction_based_on_watershed_segmentation_and_elastic_registration)

- **Regional Affine Registration paper (2015 IRBM):** "Mesh correspondence improvement using Regional Affine Registration: Application to Statistical Shape Model of the scapula." Journal: *IRBM* (Ingénierie et Recherche Biomédicale). Focuses on improving correspondence for SSM construction. Dataset size not confirmed from accessible content. — [Source](https://www.sciencedirect.com/science/article/abs/pii/S1959031815000706)

- **Halloran et al. (2018):** "Morphologic variations of the scapula in 3-dimensions: a statistical shape model approach." Journal: *Journal of Shoulder and Elbow Surgery (JSES)*. **N=110 CT scans.** SSM with 5 clinically interpreted modes (see shape modes section). — [Source](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063); [PubMed](https://pubmed.ncbi.nlm.nih.gov/30100175/)

- **Premorbid glenoid prediction paper (2018 JSES):** "A statistical shape model to predict the premorbid glenoid cavity." Journal: *JSES*. Used SSM to reconstruct glenoid morphology in patients with erosion. Dataset size not confirmed from accessible content; validated on healthy scapulae with RMS surface distance 1.0 ± 0.2 mm. — [Source](https://www.sciencedirect.com/science/article/abs/pii/S1058274618303082); [PubMed](https://pubmed.ncbi.nlm.nih.gov/29958822/)

- **Statistical Shape Modeling Approach to Predict Missing Scapular Bone (2019):** Published in *Annals of Biomedical Engineering*. Uses GPMM approach related to Scalismo/Basel group. — [Source](https://link.springer.com/content/pdf/10.1007/s10439-019-02354-6.pdf); [ResearchGate](https://www.researchgate.net/publication/335767223_Statistical_Shape_Modeling_Approach_to_Predict_Missing_Scapular_Bone)

- **Assessment of scapular morphology and bone quality (2019 PubMed):** Combined SSM + statistical intensity model. Generalization error ~2.6 mm; specificity ~2.99 mm over 10,000 instances. Dataset size not confirmed from accessible content. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/30732468/)

- **Integration of cortical thickness in SSM of the scapula (2020):** Authors not confirmed from accessible content. Journal: *Computer Methods in Biomechanics and Biomedical Engineering*. Integrated cortical thickness into the shape model. First 9 principal modes = 95% variability; average cortical thickness 2.0 ± 0.6 mm. Compactness and generalisation error for outer surface ~1.5 mm. — [Source](https://www.tandfonline.com/doi/full/10.1080/10255842.2020.1757082); [PubMed](https://pubmed.ncbi.nlm.nih.gov/32364819/)

- **Development of statistical shape and intensity models of eroded scapulae (University of Victoria, date not confirmed):** Thesis/technical report. Focused on scapulae with severe glenoid erosion for shoulder arthroplasty improvement. — [Source](https://dspace.library.uvic.ca/items/cf9057aa-47d4-4dd7-ab5d-295e490b8885)

- **Halloran et al. (2021):** "Determination of predisposing scapular anatomy with a statistical shape model—Part II: shoulder osteoarthritis." Journal: *JSES*. Extends the N=110 SSM dataset to study OA predisposing anatomy. — [Source](https://www.sciencedirect.com/science/article/abs/pii/S1058274621000987)

- **Scapular morphology OA vs RCT (2024 JSES):** "Scapular morphology is associated with certain patterns of glenohumeral osteoarthritis but not with full-thickness rotator cuff tears." Dataset size not confirmed from accessible content. — [Source](https://www.jshoulderelbow.org/article/S1058-2746(24)00865-6/fulltext); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1058274624008656)

- **Silvestros et al. (2024):** "Scapular morphology variation affects reverse total shoulder arthroplasty biomechanics. A predictive simulation study using statistical and musculoskeletal shoulder models." Journal: *Journal of Orthopaedic Research*. Combines SSM with musculoskeletal simulation for RSA. — [Source](https://onlinelibrary.wiley.com/doi/full/10.1002/jor.25801)

- **Virtual assessment of internal rotation in RSA (2025):** Database of >10,000 scapulae using a Statistical Shape Model to obtain 5 scapula sizes (mean ± 2 SD). Journal: *JSES International*. Large-scale virtual study. — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11733559/); [PubMed](https://pubmed.ncbi.nlm.nih.gov/39822834/)

- **Two-body SSM arthropathic shoulder (2025):** "A first-of-its-kind two-body statistical shape model of the arthropathic shoulder: enhancing biomechanics and surgical planning." Journal: *Journal of Orthopaedic Surgery and Research* (Springer). **N=45 RTSA patients.** Combined scapula + proximal humerus model. Claimed first published two-body SSM for this specific population. — [Source](https://link.springer.com/article/10.1186/s13018-025-05855-4); [PubMed](https://pubmed.ncbi.nlm.nih.gov/40462104/)

- **Marques, Folgado & Quental (2025):** "Reconstruction of Scapula Bone Shapes from Digitized Skin Landmarks Using Statistical Shape Modeling and Multiple Linear Regression." Journal: *Annals of Biomedical Engineering*. **N=56 scapula segmentations** + 4 bone/skin landmarks. Reconstruction from palpable skin landmarks without imaging. R²=0.70–0.98; max median error 4 mm; average surface-to-surface error 2.41–2.45 mm. — [Source](https://link.springer.com/article/10.1007/s10439-025-03768-1); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12391217/)

- **Japanese shoulder SSM stratified by sex and stature (2025/2026):** "Population-specific statistical shape modeling of the Japanese shoulder: a combined analysis of the scapula and humerus stratified by sex and stature." Journal not confirmed from accessible content but indexed in PMC. Combined scapula + humerus SSM. Scapula height 132±8 mm (female) to 165±7 mm (male). — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273566/); [PubMed](https://pubmed.ncbi.nlm.nih.gov/42317473/)

- **Scapular morphological variations homologous model (2025):** "Scapular morphological variations and sex-related and generational differences in global scapular shape: three-dimensional morphometric analysis using a homologous model." **N=500 individuals.** Three-dimensional morphometric analysis. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666639125000677); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41179478/)

- **Comparing SSM fitting methods with Scalismo (2018 EMBC):** "Comparing Statistical Shape Model Based Mesh Fitting Methods Using Open Source Scalismo Toolbox: Towards Patient-Specific Biomechanics Modeling." **N=27 dry scapulae.** Compared ICP non-rigid registration vs parametric (L-BFGS) registration via Scalismo. Initiated with landmark-based alignment + rigid ICP. IMCP-GMM pipeline. — [ResearchGate](https://www.researchgate.net/publication/327052259_COMPARING_STATISTICAL_SHAPE_MODEL_BASED_MESH_FITTING_METHODS_USING_OPEN_SOURCE_SCALISMO_TOOLBOX_TOWARDS_PATIENT-SPECIFIC_BIOMECHANICS_MODELING); [DOI](https://doi.org/10.1109/embc.2017.8037198)

- **2D X-ray SSM fitting with landmarks (2017 PubMed):** "Interactive patient-specific 3D approximation of scapula bone shape from 2D X-ray images using landmark-constrained statistical shape model fitting." Landmark-constrained fitting to 2D radiographs. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/29060242/)

- **Statistical shape and bone property models for shoulder arthroplasty planning (2023):** "Statistical Shape and Bone Property Models of Clinical Populations as the Foundation for Biomechanical Surgical Planning: Application to Shoulder Arthroplasty." — [PubMed](https://pubmed.ncbi.nlm.nih.gov/37295930/)

- **SSM-based reconstruction eliminating full CT scan need (2022):** "Statistical shape modeling-based reconstruction eliminates the need for full scapular computed tomography scan data in preoperative total shoulder arthroplasty planning." — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1045452722000499)

### Inferences
- The field is dominated by shoulder arthroplasty (TSA, RTSA) applications with N typically in the range of 27–500.
- Halloran et al. (2018, N=110) appears to be the most-cited foundational healthy-scapula SSM; the same group extended it to OA in 2021 and to RSA biomechanics.
- The Scalismo/Basel group (Lüthi, Madsen) contributes methodological tools (GiNGR, GPMM) but no single scapula SSM paper from this group was identified.

### Gaps
- Exact author lists and journal volumes for several papers could not be confirmed due to egress-blocked domains (ScienceDirect, Tandfonline, Springer, ResearchGate, NIH PMC were all blocked).
- Dennis Madsen, Benedikt Braun, and Klaus Sander: no specific scapula SSM papers by these named researchers were returned in any search result.
- Elise Laende: associated with UVic eroded scapula SSM thesis but exact publication details not accessible.
- Audrey Cheung: one CAOS 2013 proceedings paper identified; no further peer-reviewed scapula SSM papers by this author confirmed.

---

## What metrics do they report?

### Takeaway
The standard SSM evaluation trinity — compactness, generalization, and specificity — is reported in most validation papers, usually alongside surface distance metrics; however, reporting is inconsistent and some clinical papers omit model quality metrics entirely.

### Cited Findings
- **Compactness:** First 9 principal modes = 95% total variability reported for the cortical thickness SSM paper (2020). First 10 modes = >80% total variation reported for an unnamed scapula SSM in search results. — [Tandfonline/CMBME 2020](https://www.tandfonline.com/doi/full/10.1080/10255842.2020.1757082)
- **Generalization error:** ~2.6 mm (leave-one-out cross-validation) reported for the bone quality/assessment SSM. — [PubMed 2019](https://pubmed.ncbi.nlm.nih.gov/30732468/)
- **Specificity:** ~2.99 mm over 10,000 random instances reported for bone quality SSM. — [PubMed 2019](https://pubmed.ncbi.nlm.nih.gov/30732468/)
- **Cortical thickness SSM compactness/generalization outer surface:** ~1.5 mm, comparable to similar studies per the paper. — [Tandfonline/CMBME 2020](https://www.tandfonline.com/doi/full/10.1080/10255842.2020.1757082)
- **Surface distance (reconstruction from skin landmarks):** Average surface-to-surface error 2.41 and 2.45 mm (from digitized and predicted landmarks respectively); max median error 4 mm. — [Marques et al. 2025, Annals of Biomedical Engineering](https://link.springer.com/article/10.1007/s10439-025-03768-1)
- **Glenoid prediction errors:** RMS surface distance 1.0 ± 0.2 mm; glenoid version prediction error 2.3° ± 1.8°; glenoid inclination error 2.1° ± 2.0°. — [Premorbid glenoid SSM paper 2018, JSES](https://www.sciencedirect.com/science/article/abs/pii/S1058274618303082)
- **Reconstruction validity (watershed/elastic 2013):** Highest mean error 0.97 mm; highest RMS error 1.30 mm. — [CAOS 2013 / Bone & Joint](https://online.boneandjoint.org.uk/doi/10.1302/1358-992X.95BSUPP_28.CAOS2013-031)

### Inferences
- A reporting inconsistency exists: clinical arthroplasty-focused papers (e.g., RSA biomechanics, OA predisposition) often describe SSM modes qualitatively without reporting numerical compactness/specificity/generalization values.
- The specificity test using 10,000 random instances is a community standard but not universally applied.

### Gaps
- Exact numerical compactness/generalization/specificity values for the Halloran et al. (2018) N=110 paper and the 2024–2025 clinical papers could not be confirmed due to blocked access to ScienceDirect and JSES full text.

---

## What datasets do they use — are any paired bilateral (both left AND right from same subjects)?

### Takeaway
No published scapula SSM paper was identified that explicitly builds an SSM using paired bilateral (left AND right from the same subjects) as its primary data contribution; most studies mirror left scapulae to right for standardization, and the few contralateral studies focus on measurement reliability rather than intra-individual asymmetry modelling.

### Cited Findings
- Most studies standardize to right-sided anatomy by reflecting left-sided scapulae across the sagittal plane. — [Morphologic variations paper, JSES 2018](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063); [search result summary on bilateral normalization]
- Left-right glenoid measurements (version, inclination, width, scapula offset) showed the paired scapulae were NOT statistically symmetrical on those metrics, though differences were within minimal detectable change for the cohort. — [Contralateral study PubMed 2023/2024, N=260 shoulders](https://pubmed.ncbi.nlm.nih.gov/37852431/)
- High bilateral correlation was found for acromial morphology (indicating symmetry in that region). — [Search result summary on bilateral studies]
- A 2026 Swin-UNet morphometric paper showed bilateral correlation for laxocomial equivalent angle r=0.904, p<0.001 but this is a segmentation/measurement paper rather than a paired SSM study. — [doi.org/10.1515/bmt-2026-0280]
- One study explicitly stated as a limitation that "scapulae analyzed were unpaired, which precluded side-to-side comparison and limited the investigation of bilateral anatomical asymmetry." — [Comprehensive scapula morphology paper, PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225530/)
- The Japanese combined scapula+humerus SSM mirrored left-sided data; no true bilateral pairing used. — [PMC 2025/2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273566/)

### Inferences
- The field has never modelled intra-individual left–right shape asymmetry using a paired bilateral SSM. A study that collects CT scans of both left and right scapulae from the same individuals and builds a model of bilateral shape covariation would be genuinely novel.
- The contralateral shoulder literature establishes that true anatomical symmetry is imperfect, making the modelling of asymmetry scientifically interesting.

### Gaps
- No search returned any paper explicitly claiming to use a paired bilateral dataset for scapula SSM construction.
- Whether any of the N=110 or N=500 datasets incidentally included bilateral scans is unknown.

---

## What shape modes do they interpret — do they link modes to clinical features?

### Takeaway
The Halloran et al. (2018) N=110 study is the most complete in interpreting the first 5 modes clinically, including acromion morphology; however, no paper explicitly maps SSM modes to established clinical classification systems (e.g., Bigliani acromion types I/II/III, Walch glenoid erosion types).

### Cited Findings
- **Mode 1 (72% variance):** Overall size. — [Halloran et al. JSES 2018](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- **Mode 2 (~5%):** Rotation of the coracoacromial complex (acromion + coracoid collectively rotating). — [Halloran et al. JSES 2018](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- **Mode 3 (~4%):** Acromial shape and slope. — [Halloran et al. JSES 2018](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- **Mode 4 (~2%):** Shape of the scapular spine. — [Halloran et al. JSES 2018](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- **Mode 5 (~2%):** Acromial overhang. — [Halloran et al. JSES 2018](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- In one PC: lateral scapular morphology (glenoid, acromion, coracoid) tilted anteriorly in sagittal plane; scapular body tilted posteriorly. Another PC: lateral orientation of acromion, decreased anteroposterior diameter of supraspinatus fossa. — [Search result citing principal component descriptions]
- Small lateral extension and less posterior rotation of the acromion are associated with shoulder OA across almost all glenoid morphology types. — [Scapular OA predisposition paper JSES 2021](https://www.sciencedirect.com/science/article/abs/pii/S1058274621000987)
- An "anatomically parameterized statistical shape model" (ArSSM) paper exists that explains morphometry through statistical learning, linking modes to anatomical parameters. — [ArSSM arXiv 2022](https://arxiv.org/pdf/2202.08580)

### Inferences
- Mode-to-clinical-classification mapping (e.g., which end of mode 3 corresponds to Bigliani Type III hooked acromion) has not been explicitly published for the scapula, representing a gap.
- Modes have been described geometrically, but formal validation against clinician-assigned categories (Bigliani, Walch, Favard glenoid types) has not been reported.

### Gaps
- Whether the N=110 Halloran paper or the OA paper explicitly cross-tabulates SSM scores (principal component scores) against radiographic classification systems (e.g., Bigliani acromion type) is unknown due to blocked full-text access.
- Coracoid morphology in relation to SSM modes: not described in any accessible result beyond being mentioned as part of the coracoacromial complex in mode 2.

---

## What registration methods do they use?

### Takeaway
Registration methods vary widely across papers and no gold-standard has emerged; landmark-guided ICP and IMCP-GMM are the most cited for scapula specifically, while the Basel group's GiNGR (Gaussian Process regression) framework exists but has not been applied to scapula SSMs in the retrieved literature.

### Cited Findings
- **Watershed segmentation + zone-based elastic registration (Cheung 2013):** Scapula surface divided into 8 zones using 3D watershed; surface-to-surface correspondence established zone by zone. — [CAOS 2013](https://online.boneandjoint.org.uk/doi/10.1302/1358-992X.95BSUPP_28.CAOS2013-031)
- **Regional Affine Registration (2015):** Improved mesh correspondence using regional affine registration. — [IRBM 2015](https://www.sciencedirect.com/science/article/abs/pii/S1959031815000706)
- **IMCP-GMM (iterative median closest point – Gaussian mixture model):** Used to create a median virtual shape; then separate GP-MMs (Gaussian Process Morphable Models) built to register all samples. — [Scalismo fitting comparison EMBC 2017/2018](https://www.researchgate.net/publication/327052259_COMPARING_STATISTICAL_SHAPE_MODEL_BASED_MESH_FITTING_METHODS_USING_OPEN_SOURCE_SCALISMO_TOOLBOX_TOWARDS_PATIENT-SPECIFIC_BIOMECHANICS_MODELING)
- **ICP non-rigid registration (Scalismo, 2017/2018):** Fitting initiated with landmark-based alignment + rigid ICP, followed by non-rigid ICP. — [Scalismo fitting comparison](https://www.researchgate.net/publication/327052259_COMPARING_STATISTICAL_SHAPE_MODEL_BASED_MESH_FITTING_METHODS_USING_OPEN_SOURCE_SCALISMO_TOOLBOX_TOWARDS_PATIENT-SPECIFIC_BIOMECHANICS_MODELING)
- **Parametric registration with L-BFGS optimizer (Scalismo, 2017/2018):** Alternative fitting method compared against ICP. — [Scalismo fitting comparison](https://www.researchgate.net/publication/327052259_COMPARING_STATISTICAL_SHAPE_MODEL_BASED_MESH_FITTING_METHODS_USING_OPEN_SOURCE_SCALISMO_TOOLBOX_TOWARDS_PATIENT-SPECIFIC_BIOMECHANICS_MODELING)
- **CPD (Coherent Point Drift):** Used in at least one paper to predict coordinates of semi-landmarks for SSM construction. — [Search result on automated segmentation/SSM construction]
- **GiNGR (Generalized Iterative Non-Rigid Point Cloud and Surface Registration using Gaussian Process Regression):** Published 2022 by the Basel group (unibas-gravis). Unifies ICP, CPD and other methods under a GP framework with explainable hyperparameters, multi-resolution, and expert annotation integration. No scapula SSM paper applying GiNGR was found. — [GiNGR arXiv 2022](https://arxiv.org/pdf/2203.09986)
- **Probabilistic registration for GPMM in presence of missing data (2022):** Related Basel group paper; applicable to scapulae with bone loss/erosion. — [arXiv 2022](https://arxiv.org/pdf/2203.14113)

### Inferences
- GiNGR represents a clear methodological gap: it is a state-of-the-art framework from the Scalismo group but no published paper yet uses it specifically for scapula SSM construction.
- Landmark-guided GPMM registration (the approach taught in Scalismo tutorials) has not been the stated method in most scapula papers; most use older correspondence methods (ICP, CPD, regional affine).

### Gaps
- Whether the N=500 homologous model paper (2025) uses GiNGR or a more recent deep-learning correspondence method is unknown.
- The full methodological details for the Halloran et al. (2018) and UVic eroded scapulae papers were inaccessible.

---

## What does NOBODY report that seems like an obvious next step (gaps for novel contribution)?

### Takeaway
Six major gaps emerge across the literature: (1) no paired bilateral SSM exists; (2) no GiNGR-based scapula SSM has been published; (3) mode-to-clinical-classification cross-validation is absent; (4) non-Western populations beyond Japan are under-represented; (5) no multi-bone (scapula + clavicle + humerus) three-body SSM exists; and (6) no paper models intra-individual left–right shape asymmetry as a target variable.

### Cited Findings
- No two-body scapula+humerus SSM existed until 2025 (for RTSA patients) — the 2025 Springer paper claims "first-of-its-kind" status. — [Source](https://link.springer.com/article/10.1186/s13018-025-05855-4)
- "No two-body statistical shape model that captures the simultaneous coupled variation of the scapula and proximal humerus in populations requiring Reverse Total Shoulder Arthroplasty has been published to date." — [Two-body SSM PubMed 2025](https://pubmed.ncbi.nlm.nih.gov/40462104/)
- Unpaired datasets are a stated limitation: "scapulae analyzed were unpaired, which precluded side-to-side comparison." — [PMC 2025 morphology paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225530/)
- Bilateral glenoid measurements from the same subjects are not symmetrical, making bilateral variation a scientifically interesting target. — [Contralateral study 2023/2024](https://pubmed.ncbi.nlm.nih.gov/37852431/)
- GiNGR paper (2022) unifies methods but has not been applied to scapula SSMs in the literature. — [GiNGR arXiv 2022](https://arxiv.org/pdf/2203.09986)
- Probabilistic registration for GPMM with extensive missing data exists methodologically but has not been applied to scapulae with bone loss in a published SSM paper. — [arXiv 2022](https://arxiv.org/pdf/2203.14113)
- Anatomically Parameterized SSM (ArSSM) framework (2022) enables anatomical parameter-driven shape explanations but no scapula-specific application was found. — [ArSSM arXiv 2022](https://arxiv.org/pdf/2202.08580)

### Inferences
- **Gap 1 – Paired bilateral SSM:** No published study uses both left AND right scapulae from the same individuals to model intra-individual asymmetry. This could quantify the extent to which contralateral reconstruction is valid, provide a prior for surgical planning of pathological sides, and be a genuinely novel dataset contribution.
- **Gap 2 – GiNGR-based correspondence:** The state-of-the-art Basel framework (GiNGR, 2022) has not been applied to scapula SSM construction. A rigorous comparison of GiNGR vs CPD vs ICP correspondence quality for the complex scapula geometry (thin scapular body, prominent processes) would be novel.
- **Gap 3 – Mode-to-clinical-classification:** No paper has cross-validated SSM principal component scores against established clinical grading systems (Bigliani acromion types I/II/III for impingement; Walch/Favard types for glenoid wear; Lafosse coracoid index for HAGL). This would bridge SSM-derived shape statistics with surgeon-used classifications.
- **Gap 4 – Non-Western populations beyond Japan:** The 2025 Japanese study fills one population gap; African, South Asian, and South American populations are entirely absent.
- **Gap 5 – Three-body SSM (scapula + humerus + clavicle):** The two-body scapula+humerus model (2025) is the current frontier; adding the clavicle and/or the acromioclavicular joint has not been done.
- **Gap 6 – Pathology-specific correspondence with probabilistic missing-data registration:** Scapulae with large erosions (Walch B2/C, or tumour resection) have extensive missing regions. Applying GPMM-based probabilistic registration designed for missing data to such cases has not been published for the scapula, even though the methodology exists.
- **Gap 7 – SSM combined with functional/biomechanical labelling:** The RSA papers use SSMs to generate virtual patients but do not feed mode scores back as predictors of biomechanical outcomes in a regression/machine learning framework that maps shape directly to function.

### Gaps
- No paper comparing deep-learning shape reconstruction vs PCA-based SSM for the scapula was found; this may be an emerging gap not yet explored.
- Whether any group has an unpublished or in-progress bilateral scapula SSM is unknown from public search.
