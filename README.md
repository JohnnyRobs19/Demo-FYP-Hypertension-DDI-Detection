# Machine Learning-Driven Prediction for Safe Medication Pathways among Hypertensive Patients

Final Year Project - University of Malaya

Student: Jonathan Siew Zunxian
Supervisor: Dr. Unaizah Hanum Obeidellah
Collaborator: Dr. Nurulhuda Abdul Manaf and Dr. Nur Aishah Che Roos (UPNM)

---

## Problem Statement

Hypertension patients in Malaysia are often prescribed multiple medications simultaneously, increasing the risk of drug-drug interactions (DDIs). Despite the availability of Malaysian Clinical Practice Guidelines (CPGs), there is limited systematic analysis of prescribing patterns and DDI risks in polypharmacy scenarios. This project addresses the need for automated DDI screening and evidence-based medication pathway recommendations.

---

## Objectives

1. Identify common drug classes prescribed to hypertension patients
2. Assess the prevalence of potential DDIs using standard interaction checkers
3. Propose safer medication pathways through ML-driven predictions and knowledge-driven recommendations

---

## Methodology

The project implements three core modules:

### Module 1: Automated Data Collection

**Purpose:** Collect DDI severity data for 406 drug pair combinations from DrugBank

**Technical approach:**
- Playwright-based web automation for asynchronous data collection
- Fuzzy matching algorithm for dropdown element selection
- Checkpoint-based resumption system for interrupted scraping sessions

**Files:** `drugbank_ddi_scraper.py`, `debug_drugbank_page_load.py`, `debug_drugbank_add_drugs.py`, `debug_drugbank_html_analysis.py`

### Module 2: Knowledge-Driven Explainability Framework

**Purpose:** Enrich ML predictions with clinical context from meta-analyses

**Clinical rules implemented:**
- Rule A: ACEI vs ARB mortality benefit (Alcocer 2023)
- Rule B: ACEI tolerability and cough risk (Hu 2023)
- Rule C: CCB+RAAS combination therapy for edema reduction (Makani 2011)
- Rule D: Diuretic efficacy comparison (Roush 2015)
- Rule E: Beta-blocker phenotype targeting (Mahfoud 2024)

**Files:** `add_xai_framework.py`, `create_sample_drug_pairs.py`

### Module 3: Machine Learning Prediction

**Purpose:** Predict DDI severity for drug combinations

**Technical approach:**
- Algorithm: Random Forest classifier
- Validation: 5-fold cross-validation
- Hyperparameter optimization: GridSearchCV
- Class imbalance handling: Weighted learning

**Files:** `Random_Forest_DDI_Analysis_and_Training_DrugBank_Only.ipynb`, `FYP_DrugBank_Inclusive.csv`

### Module 4: Mathematical Pathway Ranking

**Purpose:** Rank medication pathways using a mathematical framework

**Technical approach:**
- Four components: Interaction complexity, safety floor, foundation constraint, specialist score
- Lexicographic ordering for pathway ranking
- CPG compliance evaluation

**Files:** `mathematical_pathway_simulator.py`

---

## Technical Implementation

### Data Collection Pipeline
- Asynchronous web scraping with Playwright
- Anti-detection measures for stable operation
- Checkpoint system for robustness

### XAI Framework
- Pandas vectorized operations for rule application
- Coverage: 378 out of 406 drug pairs (93.1%)
- Integration of 5 evidence-based clinical rules

### Machine Learning Model
- Nested cross-validation for model evaluation
- Hyperparameter search space: 216 configurations
- Optimal configuration: 300 trees, max depth 15, balanced class weights

### Database Operations
- CSV-based data storage and retrieval
- Create, read, update operations for datasets
- Checkpoint tracking for scraping progress

---

## Results

### Drug Class Analysis
- 5 drug classes analyzed: ACEI, ARB, CCB, Beta-Blockers, Diuretics
- 29 medications from Malaysian CPG
- 406 unique drug pair combinations

### DDI Prevalence
- Severity distribution: Major (2%), Moderate (52.2%), Minor (35%), No Interaction (10.8%)
- Most common interactions: Beta-Blocker + CCB, ACEI + ARB

### Model Performance
- Accuracy: 77.59% (±6.10%)
- ROC AUC: 89.08% (±6.87%)
- Major DDI Recall: 80.00% (±24.49%)

### XAI Integration
- 378 drug pairs enriched with clinical rules
- Evidence-based recommendations for safer pathways

---

## File Structure

```
Data Collection
├── drugbank_ddi_scraper.py
├── debug_drugbank_page_load.py
├── debug_drugbank_add_drugs.py
└── debug_drugbank_html_analysis.py

Knowledge Integration
├── add_xai_framework.py
└── create_sample_drug_pairs.py

Machine Learning
├── Random_Forest_DDI_Analysis_and_Training_DrugBank_Only.ipynb
└── FYP_DrugBank_Inclusive.csv

Mathematical Framework
└── mathematical_pathway_simulator.py

Documentation
└── README.md
```

---

## Technologies Used

**Web Scraping:** Playwright, BeautifulSoup4, asyncio

**Data Processing:** pandas, NumPy

**Machine Learning:** scikit-learn (RandomForestClassifier, GridSearchCV, StratifiedKFold)

**Visualization:** matplotlib, seaborn

**Development:** Python 3.12, Jupyter Notebook, Git

---

## Usage

### Data Collection
```bash
python drugbank_ddi_scraper.py demo_pairs.csv --debug
```

### XAI Framework
```bash
python add_xai_framework.py
```

### Machine Learning Training
```bash
jupyter notebook Random_Forest_DDI_Analysis_and_Training_DrugBank_Only.ipynb
```

### Mathematical Pathway Simulator
```bash
python mathematical_pathway_simulator.py
```

---

## Clinical Evidence Base

XAI rules are based on peer-reviewed meta-analyses:

- Alcocer et al. (2023): ACEI vs ARB mortality outcomes
- Hu et al. (2023): ACEI tolerability profile
- Makani et al. (2011): CCB+RAAS combination therapy
- Roush et al. (2015): Diuretic comparative efficacy
- Mahfoud et al. (2024): Beta-blocker patient selection

---

## Limitations

- Dataset limited to 29 Malaysian CPG-approved medications
- DDI data sourced from single database (DrugBank)
- Model validation performed on same dataset used for training
- Clinical recommendations require validation by medical professionals

---

## Contact

Jonathan Siew Zunxian
Faculty of Computer Science & Information Technology
University of Malaya

---

## License

This project is developed as part of academic requirements at University of Malaya.
For academic and educational purposes only.
