# Bilateral Bone Symmetry/Asymmetry in Statistical Shape Models

## Key Question 1: Has any SSM paper explicitly separated within-subject bilateral asymmetry from between-subject population variation using a paired dataset?

### Takeaway
No study has built an SSM that explicitly decomposes shape variance into a within-subject bilateral asymmetry component versus a between-subject population component for shoulder girdle bones; the closest work uses ICC on ankle/hindfoot bones (Tümer 2019) or combines left-right femur models to isolate positioning artefacts (Lindner 2014), but neither frames this as a formal variance-decomposition in the SSM sense.

### Cited Findings
- Tümer et al. (2019) built separate SSMs for the fibula, tibia, calcaneus, and talus from bilateral CT scans of the same subjects and used intra-class correlation (ICC) to quantify whether intra-subject shape variation was smaller than inter-subject variation for each mode of shape variation; an ICC of 1 implies all variation is inter-subject (perfect bilateral symmetry), while ICC = 0 implies intra-subject ≈ inter-subject variation — [Tümer 2019, J. Anatomy](https://onlinelibrary.wiley.com/doi/full/10.1111/joa.12900)
- Tümer 2019 found that for tibial tuberosity diameter and fibula shaft curvature, intra-subject (bilateral) variation was as large as inter-subject variation (ICC near 0), meaning symmetry cannot be assumed for those shape modes — [Tümer 2019, J. Anatomy](https://onlinelibrary.wiley.com/doi/full/10.1111/joa.12900)
- Lindner et al. (2014) constructed a combined statistical shape model of left AND right proximal femurs from 1258 Caucasian women (Osteoarthritis Initiative) to identify shape variation attributable to subject positioning, and also built a single SSM to analyze true bilateral symmetry separately — [Lindner 2014, via PMC3968883](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3968883/)
- Deschênes & Drapeau (2025) introduced a method specifically designed to segregate bilateral (within-subject) variation from interindividual (between-subject) variation for whole 3D bone models, demonstrated on the humerus and second metacarpal; they showed bilateral variation falls between interscan (lowest) and interindividual (greatest) — [Deschênes & Drapeau 2025, Am. J. Biological Anthropology](https://onlinelibrary.wiley.com/doi/10.1002/ajpa.70004)
- A multi-level SSM framework has been proposed to capture shape variation both between and within individuals (treating inter-patient features as fixed effects and intra-patient as random effects), but applications to bilateral bones are not reported in the orthopaedic literature — [Frontiers multi-level shape models 2023](https://www.frontiersin.org/articles/10.3389/fbioe.2023.1089113/full)
- The geomorph R package `bilat.symmetry` function implements Procrustes ANOVA that decomposes shape variation into: (i) among individuals, (ii) directional asymmetry (side), and (iii) fluctuating asymmetry (individual × side interaction) — [geomorph package docs](https://rdrr.io/github/EmSherratt/geomorph/man/bilat.symmetry.html)

### Inferences
- The ICC approach used by Tümer 2019 is mathematically equivalent to a one-way random-effects model (subject as random factor, side nested within subject), which is a simplified version of the within/between decomposition; it stops short of full variance-component estimation in the SSM (PCA) sense.
- No study appears to have applied the Procrustes ANOVA bilateral decomposition within the context of building a population-level SSM (i.e., using the decomposition to understand how much of the principal component variance reflects true inter-subject biology vs. within-subject asymmetry noise).

### Gaps
- No published paper was found that builds an SSM using paired bilateral data from the same subjects AND formally partitions total shape variance into (a) between-subject population variation and (b) within-subject bilateral asymmetry components in the SSM/PCA framework — this appears to be a genuine methodological gap.
- No studies on paired bilateral scapula or shoulder girdle datasets performing this decomposition were found.


## Key Question 2: What is known about bilateral symmetry of the scapula specifically — are the two sides statistically different in 3D shape?

### Takeaway
Paired left-right scapulae show statistically significant differences in glenoid version, inclination, width, and scapula offset, but the mean bilateral differences (≤2° and ≤2 mm) are within the minimal detectable change for the cohort, suggesting clinical symmetry despite statistical asymmetry.

### Cited Findings
- A study of 130 bilateral shoulder pairs (260 shoulders, CT-based) found statistically significant differences between right and left glenoid version (−5.3° vs. −4.6°), inclination (8.4° vs. 9.3°), and width (25.6 mm vs. 25.4 mm), as well as scapula offset (105.8 mm vs. 106.2 mm); however, mean bilateral differences were only ~2° and ~2 mm, within the minimal detectable change — [Premorbid glenoid anatomy, JSES 2024, PubMed 37852431](https://pubmed.ncbi.nlm.nih.gov/37852431/)
- The Mouffoke et al. (2018) scapula SSM used bilateral CT from 54 patients (108 shoulders) but mirrored all left-sided scapulae to right-sided equivalents before building the SSM; it did not analyze bilateral asymmetry explicitly as a component of variation — [Mouffoke 2018, J. Shoulder Elbow Surg.](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- The first 5 shape modes from the Mouffoke 2018 scapula SSM accounted for: size (72%), coracoacromial complex rotation (5%), acromial shape and slope (4%), scapular spine shape (2%), and acromial overhang (2%); because left-side scapulae were mirrored rather than modelled as paired, bilateral asymmetry is not independently captured as a mode — [Mouffoke 2018, J. Shoulder Elbow Surg.](https://www.sciencedirect.com/science/article/abs/pii/S1058274618304063)
- A 2025 population-specific SSM of the Japanese shoulder (combined scapula + humerus) stratified by sex and stature was published, but no explicit bilateral decomposition was performed — [PMC13273566](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273566/)
- Scapular motion asymmetry (3D kinematics, not static shape) has been assessed in participants with and without shoulder impingement syndrome — [ScienceDirect scapular asymmetry kinematics](https://www.sciencedirect.com/science/article/abs/pii/S0268003316301292)

### Inferences
- The practice of mirroring one side to align all bones to the same anatomical orientation (used in Mouffoke 2018 and many other SSM studies) means the resulting SSM conflates true bilateral shape symmetry with the modelling assumption of symmetry — any bilateral difference is averaged away rather than measured.
- The statistical significance of the glenoid measurements across 130 pairs is likely driven by the large sample size; the clinical relevance of ≤2° and ≤2 mm differences is debated.

### Gaps
- No study has quantified 3D whole-surface bilateral shape difference for the scapula using dense surface correspondence (as opposed to a handful of geometric measurements); a surface-to-surface distance map or Hausdorff distance analysis of paired scapulae is apparently absent in the literature.
- No study has reported the percentage of total scapula shape variance attributable to bilateral asymmetry vs. between-subject variation using formal variance components.


## Key Question 3: For other paired bones (femur, tibia, humerus), have SSM papers used paired data to study asymmetry — and what did they find?

### Takeaway
Several studies have used paired bilateral data for the femur, tibia/fibula/calcaneus/talus, and humerus; the general finding is that bilateral variation is smaller than inter-subject variation for most shape modes, but for specific anatomical features (fibula shaft curvature, tibial tuberosity, humerus epiphyses) intra-subject bilateral variation can approach inter-subject levels.

### Cited Findings
- Tümer 2019 (fibula, tibia, calcaneus, talus): used bilateral CT from the same subjects; ICC indicated near-perfect symmetry (ICC ≈ 1) for most modes, but tibial tuberosity diameter and fibula shaft curvature showed ICC near 0 (bilateral ≈ inter-individual variation) — [J. Anatomy 2019, PMC6284442](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6284442/)
- Lindner et al. 2014 (proximal femur): combined SSM of left AND right femurs from 1258 subjects showed that a major source of apparent variation was subject positioning artefact (limb rotation during imaging); after adjusting for this, true bilateral asymmetry was small but detectable — [PMC3968883](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3968883/)
- Deschênes & Drapeau 2025 (humerus, second metacarpal): whole-bone bilateral asymmetry was measurable and fell between interscan variation (lowest) and interindividual variation (greatest); asymmetry was most pronounced at the radial groove, deltoid tuberosity, olecranon fossa, and epiphyses — [Am. J. Biol. Anthropology 2025](https://onlinelibrary.wiley.com/doi/10.1002/ajpa.70004)
- A 3D study of 75 distal tibia pairs reported a median left-right difference of 0.57 mm for the entire tibial plafond and 0.53 mm for the articulating surface — [PMC11666608, tibial plafond symmetry 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666608/)
- Morphometric maps of bilateral asymmetry in the humerus (2021): extracted 3D models from 102 humeri (51 pairs); asymmetry measured via cross-sectional semilandmarks; epiphyses were more asymmetrical than diaphysis — [MDPI Symmetry 2021, Morphomap paper](https://www.mdpi.com/2073-8994/13/9/1711)
- Morphological symmetry of radius and ulna (PLoS ONE 2021): investigated whether contralateral forearm bones can serve as a reliable template for the opposite side; found sufficient bilateral symmetry for clinical use — [PLoS ONE 2021, radius/ulna](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0258232)
- Gabrielli et al. 2020 (ankle and hindfoot — femur, tibia, humerus, radius): side-to-side differences in bone morphology averaged ≤0.79 mm across these upper and lower limb bones — [Foot Ankle Int. 2020, DOI 10.1177/2473011420908796](https://doi.org/10.1177/2473011420908796)
- Frontiers 2019 SSM study of skeletal anatomy for sex discrimination noted that bilateral asymmetry (left vs. right) had not previously been studied in detail for most bones, with the pelvis being one exception — [Frontiers Bioengineering Biotechnology 2019, PMC6837998](https://pmc.ncbi.nlm.nih.gov/articles/PMC6837998/)
- For femur asymmetry, it is documented that femoral torsion and neck-shaft angle can differ between sides due to limb dominance, but the magnitude is generally small compared to population-level variation — [Frontiers SSM sex discrimination 2019](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00302/full)

### Inferences
- The consistent finding across all bones studied is a hierarchy: interscan variation < bilateral (within-subject) variation < interindividual (between-subject) variation, with the gap between bilateral and interindividual being variable across bones and shape modes.
- The femur Lindner 2014 paper is the closest precedent to a methodologically rigorous within/between decomposition in the SSM context, but it was motivated by a different question (imaging positioning artefact correction) and focused on 2D radiographic shape rather than 3D surfaces.

### Gaps
- No paired bilateral SSM study exists for the scapula, clavicle, or humerus in 3D that formally partitions variance into within-subject and between-subject components.
- Whether the specific pattern (which shape modes are most vs. least symmetric) generalises from lower limb bones to shoulder girdle bones is unknown.


## Key Question 4: Is the contralateral shoulder used as a surgical template for shoulder replacement? What is the evidence for/against this practice?

### Takeaway
Contralateral shoulder templating for shoulder replacement (both anatomic TSA and reverse TSA) is used in clinical practice and is considered valid given that mean bilateral glenoid differences are small (~2° version, ~2° inclination, ~2 mm width); however, formal prospective validation with a control group is lacking, and statistically significant differences between sides do exist.

### Cited Findings
- A 2023 technique article described contralateral preoperative CT templating for fracture reverse total shoulder arthroplasty; the authors acknowledged no prior studies had formally evaluated this technique, and limitations include unaccounted anatomic differences between sides — [PMC10426702, contralateral rTSA templating](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426702/)
- The 130-pair (260-shoulder) CT study confirmed that while mean bilateral glenoid differences are ≤2° and ≤2 mm (within minimal detectable change), statistically significant differences exist; the authors concluded that healthy contralateral shoulders can be useful templates for TSA planning — [JSES 2024, PubMed 37852431](https://pubmed.ncbi.nlm.nih.gov/37852431/)
- Some surgical technique guidelines state: "if there is marked deformation of the head, planning should be based on the healthy contralateral joint" — [Zimmer Biomet Anatomical Shoulder Surgical Technique PDF](https://www.zimmerbiomet.com/content/dam/zimmer-biomet/medical-professionals/000-surgical-techniques/shoulder/anatomical-shoulder-system-surgical-technique.pdf)
- SSM-based premorbid glenoid reconstruction (as an alternative to contralateral templating): a 2024 study developed statistical shape model-based prediction of premorbid 3D glenoid anatomy, which may reduce reliance on the assumption of bilateral symmetry — [ScienceDirect 2024, premorbid glenoid SSM](https://www.sciencedirect.com/science/article/abs/pii/S1045452724000580)
- For hip arthroplasty (a parallel case), contralateral lesser trochanter is studied as a reference: a 3D analysis confirmed acceptable reliability for planning purposes — [PMC7953689](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7953689/)
- 3D templating (not contralateral) for TSA showed 100% agreement between templated and implanted glenoid prosthesis size in one series — [CIOS 2020, PubMed 32489546](https://pubmed.ncbi.nlm.nih.gov/32489546)

### Inferences
- The current clinical use of contralateral templating is supported by pragmatic evidence (bilateral differences are small), but it rests on an implicit assumption of bilateral symmetry that has not been formally tested in the SSM/shape modelling framework.
- SSM-based premorbid reconstruction methods (using population-level shape priors rather than the contralateral side) are emerging as an alternative, and their comparison to contralateral templating is an active research area.

### Gaps
- No randomised or prospective study comparing surgical outcomes with vs. without contralateral templating for shoulder arthroplasty was found.
- No study has quantified what proportion of patients have bilateral scapular asymmetry above the minimal detectable change that would make contralateral templating unreliable — the 2024 study reported means, not the distribution of outliers.


## Key Question 5: Has anyone built an SSM using BOTH sides of a paired dataset and compared it to a one-side-only SSM — does including both sides bias the model?

### Takeaway
No study has explicitly built two versions of a scapula (or shoulder) SSM — one using one side only vs. both sides — to test for bias; the practice of mirroring one side is standard but silently assumes symmetry; a 2014 femur study is the closest precedent showing that combining both sides in an SSM captures positioning artefact as a separate mode, which would otherwise inflate apparent population variation.

### Cited Findings
- Lindner et al. 2014 (proximal femur, OAI): built (a) a combined SSM of left AND right femurs and (b) a single SSM of all femurs mirrored to left, specifically to separate variation from subject positioning vs. true bilateral asymmetry; demonstrated that the combined SSM identified a major mode corresponding to positioning, not anatomy — [PMC3968883](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3968883/)
- Mouffoke et al. 2018 (scapula, 54 bilateral pairs): mirrored all left-sided scapulae to right-sided equivalents, then built a single SSM; no comparison to a bilateral-aware model was made, so whether mirroring inflated or changed the mode structure is unknown — [J. Shoulder Elbow Surg. 2018, PubMed 30100175](https://pubmed.ncbi.nlm.nih.gov/30100175/)
- A 2025 two-body SSM of the arthropathic shoulder found that combined (two-body) models outperformed single-body models for shape generation in terms of reduced bias and more plausible shapes; single-body SSMs were found to produce excessive high p-value (overly plausible) shapes — [PMC12131784, two-body shoulder SSM 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12131784/)
- Standard SSM construction practice for bilateral bones is to mirror all specimens to one canonical side before building the model; this is universal in published scapula, femur, and tibia SSMs — [Application of SSM in orthopedics review, Intelligent Medicine 2024](https://mednexus.org/doi/10.1016/j.imed.2024.05.001)
- When both sides of paired subjects are included in an SSM without accounting for pairing (treating them as independent), the effective sample size is inflated and the covariance structure is biased toward symmetry since each subject contributes two shapes that are more similar to each other than to shapes from other subjects — [Frontiers SSM sex discrimination 2019, inferred from methods discussion](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00302/full)

### Inferences
- Including both sides naively (as independent observations) in an SSM will produce a biased model: the leading principal components will partly reflect between-subject variation (correct) but some modes will reflect the repeated-measures correlation between paired sides, violating the i.i.d. assumption of PCA.
- The Lindner 2014 femur study demonstrated that deliberately including both sides can help IDENTIFY the within-subject correlation as a separate mode — a strategy that could be repurposed to study bilateral asymmetry.
- No study has explicitly shown this bias for the scapula, but the mechanism (correlation between paired observations inflating certain PCA modes) is well understood statistically.

### Gaps
- No published study has conducted a direct comparison: (a) one-sided scapula SSM vs. (b) bilateral-aware scapula SSM (accounting for paired structure), to quantify how much the mode structure and variance attribution differ.
- The question of whether models trained on both sides (naively mirrored) over- or under-estimate population shape variability compared to properly paired models has not been empirically answered for any bone in the orthopaedic SSM literature.


## Key Question 6: What statistical methods are used to test bilateral symmetry in 3D shape models (Procrustes ANOVA, mixed effects models)?

### Takeaway
The dominant methods are Procrustes ANOVA (via the geomorph R package `bilat.symmetry` function) for geometric morphometrics, ICC for comparing intra- vs. inter-subject variation in SSM modes, and surface distance metrics (mean/Hausdorff); mixed effects models in the shape space have been proposed but are rarely applied to bilateral bone comparisons.

### Cited Findings
- Procrustes ANOVA for bilateral symmetry (implemented in geomorph `bilat.symmetry`): decomposes shape variation into (i) among individuals, (ii) directional asymmetry (DA; consistent side-to-side difference), and (iii) fluctuating asymmetry (FA; individual × side interaction); uses sum-of-squared Procrustes distances as sums-of-squares — [geomorph docs](https://rdrr.io/github/EmSherratt/geomorph/man/bilat.symmetry.html); [Procrustes ANOVA Review](https://doi.org/10.1177/2473011420908796)
- `procAOVsym` in the Morpho R package provides an equivalent Procrustes ANOVA for structures with object symmetry — [Morpho package docs](https://rdrr.io/cran/Morpho/man/procAOVsym.html)
- ICC-based analysis (Tümer 2019): intra-class correlation between left and right shape coordinates per principal component mode; offers a per-mode quantification of how much of that mode's variation is within-subject (bilateral) vs. between-subject — [J. Anatomy 2019, PMC6284442](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6284442/)
- Geometric morphometrics decomposition: each shape can be decomposed into its bilaterally symmetric part (mean consensus) and its asymmetric residual; PCA on symmetric and asymmetric components separately is standard — [Bookstein/Klingenberg GM methods, as reviewed in BMC Ecol. Evol. 2011](https://link.springer.com/article/10.1186/1471-2148-11-280)
- Mixed effects SSM: a multi-subject SSM framework captures variation both between and within individuals by treating inter-patient features as fixed effects and intra-patient features as random effects; proposed for longitudinal imaging but applicable to bilateral paired data — [Frontiers multi-level SSM 2023, PMC9978224](https://pmc.ncbi.nlm.nih.gov/articles/PMC9978224/)
- Surface distance metrics: mean and max surface-to-surface distances (after rigid alignment) are used clinically to compare bilateral bone pairs; less information than full shape decomposition but widely reported — [tibial plafond symmetry, PMC11666608](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666608/)
- Assessments of bilateral asymmetry with application to human skull analysis — uses landmark-based methods and Procrustes statistics to separate DA from FA — [PMC8494363](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8494363/)
- Mardia et al. (2024) arxiv paper proposes landmark-based asymmetry analysis in the size-and-shape space, defining elementary asymmetry features at each landmark and combining into a scalar composite asymmetry measure; includes statistical tests for comparing asymmetry between groups — [arXiv 2407.17225](https://arxiv.org/abs/2407.17225)

### Inferences
- ICC per PCA mode (Tümer approach) and Procrustes ANOVA (geomorph approach) are complementary: ICC is easy to interpret per mode but assumes PCA has already been done; Procrustes ANOVA works directly on landmark/shape coordinates without requiring prior PCA.
- A full variance-component analysis for a bilateral SSM would combine: (1) compute the SSM from all subjects, both sides; (2) model shape space coordinates with a linear mixed model (subject as random, side as fixed or crossed random); (3) estimate variance components for between-subject shape variation, DA, and FA. This synthesis does not appear to have been done in published orthopaedic SSM literature.

### Gaps
- No study combining the multi-level SSM framework (mixed effects in shape space) with a paired bilateral dataset for any musculoskeletal bone was found in the orthopaedic or biomechanics literature up to 2025.
- Application of Procrustes ANOVA bilateral decomposition to SSM-derived correspondence points (as opposed to manually placed landmarks) has not been reported for any shoulder bone.
