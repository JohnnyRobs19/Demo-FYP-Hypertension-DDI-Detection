"""
================================================================================
MATHEMATICAL FRAMEWORK FOR SAFE MEDICATION PATHWAY RANKING
================================================================================
Demonstrates the complete mathematical system for evaluating and ranking
hypertension medication pathways based on four mathematical components:

1. Interaction Complexity - Using the Combination Formula C(n,2)
2. Safety Floor (s) - Minimum severity score (weakest link principle)
3. Foundation Constraint (G) - Indicator Function for CPG Rules A, B, C
4. Specialist Score (k) - Counts escalation rules (Diuretics/Beta-blockers)

FINAL RESULT: Unified Ranking Vector Q(s, G, k)
Uses Lexicographic Ordering where safety dominates, then CPG compliance, then
specialist prescription. Safety dominates because a guideline-compliant
prescription is medically invalid if it causes a life-threatening interaction.

Date: January 2026
================================================================================
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple, Set
import json

# ============================================================================
# MATHEMATICAL CONSTANTS AND DEFINITIONS
# ============================================================================

# DDI Severity Mapping (as defined in the mathematical framework)
SEVERITY_MAPPING = {
    'Major': 0,
    'Moderate': 1,
    'Minor': 2,
    'No Interaction': 3
}

# Foundation Set (CPG Compliance Rules)
FOUNDATION_SET = {'A', 'B', 'C'}

# Escalation Set (Specialist Prescription Rules)
ESCALATION_SET = {'D', 'E'}

# Drug Classification - Dynamically loaded from dataset
# This will be populated when DDISeverityCalculator is initialized
DRUG_CLASSES = {}
DRUG_TO_CLASS = {}

# ============================================================================
# COMPONENT 1 & 2: INTERACTION COMPLEXITY AND SAFETY FLOOR CALCULATOR
# ============================================================================

class DDISeverityCalculator:
    """
    Component 1: Calculates Interaction Complexity using C(n,2)
    Component 2: Calculates Safety Floor s = minimum(S) where S ∈ {0, 1, 2, 3}
    """

    def __init__(self, ddi_dataset_path: str):
        """Load the DDI dataset with severity information"""
        global DRUG_CLASSES, DRUG_TO_CLASS

        print("Loading DDI dataset...")
        self.df = pd.read_csv(ddi_dataset_path)
        print(f"✓ Loaded {len(self.df)} drug interaction pairs")

        # Build drug-to-class mapping from actual dataset
        print("Building drug classification from dataset...")
        for _, row in self.df.iterrows():
            DRUG_TO_CLASS[row['Drug_A_Name']] = row['Drug_A_Class']
            DRUG_TO_CLASS[row['Drug_B_Name']] = row['Drug_B_Class']

        # Build reverse mapping (class -> list of drugs)
        for drug, drug_class in DRUG_TO_CLASS.items():
            if drug_class not in DRUG_CLASSES:
                DRUG_CLASSES[drug_class] = []
            if drug not in DRUG_CLASSES[drug_class]:
                DRUG_CLASSES[drug_class].append(drug)

        # Sort for consistency
        for drug_class in DRUG_CLASSES:
            DRUG_CLASSES[drug_class].sort()

        print(f"✓ Loaded {len(DRUG_TO_CLASS)} unique drugs across {len(DRUG_CLASSES)} drug classes")
        for drug_class in sorted(DRUG_CLASSES.keys()):
            print(f"  • {drug_class}: {len(DRUG_CLASSES[drug_class])} drugs")

    def get_ddi_severity(self, drug_a: str, drug_b: str) -> Tuple[int, str]:
        """
        Get DDI severity between two drugs
        Returns: (severity_score, severity_label)
        """
        # Search for the interaction (bidirectional)
        interaction = self.df[
            ((self.df['Drug_A_Name'] == drug_a) & (self.df['Drug_B_Name'] == drug_b)) |
            ((self.df['Drug_A_Name'] == drug_b) & (self.df['Drug_B_Name'] == drug_a))
        ]

        if not interaction.empty:
            severity_label = interaction.iloc[0]['Final_Severity']
            severity_score = SEVERITY_MAPPING.get(severity_label, 3)
            return severity_score, severity_label
        else:
            # No interaction found - assume No Interaction
            return 3, 'No Interaction'

    def calculate_safety_floor(self, drugs: List[str]) -> Tuple[int, List[Dict]]:
        """
        Calculate Safety Floor for a pathway of n drugs

        Mathematical Formula:
        s = minimum(S) where S is the set of all pairwise DDI severities

        Returns: (safety_floor_s, interaction_details)
        """
        if len(drugs) < 2:
            raise ValueError("Pathway must contain at least 2 drugs")

        # Generate all pairwise combinations C(n, 2)
        drug_pairs = list(combinations(drugs, 2))

        interaction_details = []
        severity_scores = []

        for drug_a, drug_b in drug_pairs:
            score, label = self.get_ddi_severity(drug_a, drug_b)
            severity_scores.append(score)
            interaction_details.append({
                'pair': f"{drug_a} + {drug_b}",
                'severity_score': score,
                'severity_label': label
            })

        # Safety Floor = minimum severity score (weakest link)
        safety_floor = min(severity_scores)

        return safety_floor, interaction_details

# ============================================================================
# XAI RULE EVALUATOR (Supports Components 3 & 4)
# ============================================================================

class XAIRuleEvaluator:
    """
    Evaluates which XAI knowledge rules (A, B, C, D, E) are satisfied
    by a given medication pathway.
    Supports Component 3 (Foundation Constraint G) and Component 4 (Specialist Score k)
    """

    @staticmethod
    def evaluate_rule_a(drugs: List[str]) -> bool:
        """
        Rule A - Mortality Benefit: Pathway contains ACEI or ARB
        Clinical: ACEI/ARB provide cardiovascular mortality reduction
        """
        drug_classes = [DRUG_TO_CLASS.get(drug) for drug in drugs]
        return 'ACEI' in drug_classes or 'ARB' in drug_classes

    @staticmethod
    def evaluate_rule_b(drugs: List[str]) -> bool:
        """
        Rule B - Tolerability: Pathway includes ACEI or ARB considerations
        Clinical: ACEI (cough risk) vs ARB (better tolerability)
        """
        drug_classes = [DRUG_TO_CLASS.get(drug) for drug in drugs]
        return 'ACEI' in drug_classes or 'ARB' in drug_classes

    @staticmethod
    def evaluate_rule_c(drugs: List[str]) -> bool:
        """
        Rule C - Combination Therapy: CCB + (ACEI or ARB) combination
        Clinical: Reduces peripheral edema by 38%
        """
        drug_classes = set(DRUG_TO_CLASS.get(drug) for drug in drugs)
        has_ccb = 'CCB' in drug_classes
        has_raas = 'ACEI' in drug_classes or 'ARB' in drug_classes
        return has_ccb and has_raas

    @staticmethod
    def evaluate_rule_d(drugs: List[str]) -> bool:
        """
        Rule D - Diuretic Efficacy: Pathway contains any diuretic
        Clinical: Indapamide is superior (reduces mortality/stroke/HF), HCTZ is inferior
        """
        # Check if pathway contains any drug in the Diuretic class
        drug_classes = [DRUG_TO_CLASS.get(drug) for drug in drugs]
        return 'Diuretic' in drug_classes

    @staticmethod
    def evaluate_rule_e(drugs: List[str]) -> bool:
        """
        Rule E - Beta-Blocker Phenotype: Pathway contains beta-blocker
        Clinical: Indicated for high HR (>80 bpm) patients
        """
        drug_classes = [DRUG_TO_CLASS.get(drug) for drug in drugs]
        return 'Beta-Blocker' in drug_classes

    def evaluate_pathway(self, drugs: List[str]) -> Dict[str, bool]:
        """
        Evaluate all XAI rules for the pathway
        Returns: Dictionary of rule satisfaction
        """
        return {
            'A': self.evaluate_rule_a(drugs),
            'B': self.evaluate_rule_b(drugs),
            'C': self.evaluate_rule_c(drugs),
            'D': self.evaluate_rule_d(drugs),
            'E': self.evaluate_rule_e(drugs)
        }

# ============================================================================
# COMPONENT 3 & 4: FOUNDATION CONSTRAINT (G) AND SPECIALIST SCORE (k)
# ============================================================================

class ConstraintCalculator:
    """
    Component 3: Calculates Foundation Constraint (G) - Indicator Function
    Component 4: Calculates Specialist Score (k) - Escalation rule count
    """

    @staticmethod
    def calculate_foundation_constraint(satisfied_rules: Set[str]) -> int:
        """
        Foundation Constraint using Indicator Function:

        G = 𝟙_ℱ(P) = { 1 if ℱ ⊆ SatisfiedRules(P)
                      { 0 otherwise

        where ℱ = {A, B, C} (Foundation Set)
        """
        return 1 if FOUNDATION_SET.issubset(satisfied_rules) else 0

    @staticmethod
    def calculate_specialist_prescription(satisfied_rules: Set[str]) -> int:
        """
        Specialist Prescription (Piecewise Function):

        k = { |SatisfiedRules(P) ∩ ℰ| if (D ∈ P) ∨ (E ∈ P)
            { 0                        otherwise

        where ℰ = {D, E} (Escalation Set)
        """
        # Check if any escalation rule is in the pathway
        has_specialist = bool(satisfied_rules.intersection(ESCALATION_SET))

        if has_specialist:
            # Count how many specialist rules are satisfied (cardinality)
            k = len(satisfied_rules.intersection(ESCALATION_SET))
        else:
            k = 0

        return k

# ============================================================================
# FINAL RESULT: UNIFIED RANKING VECTOR Q(s, G, k)
# ============================================================================

class PathwayRanker:
    """
    Combines all four components into the Unified Ranking Vector Q(s, G, k).
    Uses Lexicographic Ordering where safety dominates, then CPG compliance,
    then specialist prescription.
    """

    def __init__(self, ddi_calculator: DDISeverityCalculator):
        self.ddi_calculator = ddi_calculator
        self.xai_evaluator = XAIRuleEvaluator()
        self.constraint_calculator = ConstraintCalculator()

    def evaluate_pathway(self, drugs: List[str]) -> Dict:
        """
        Complete pathway evaluation returning all components
        """
        # Step 1: Calculate Safety Floor (s)
        safety_floor, interactions = self.ddi_calculator.calculate_safety_floor(drugs)

        # Step 2: Evaluate XAI Rules
        rule_results = self.xai_evaluator.evaluate_pathway(drugs)
        satisfied_rules = {rule for rule, satisfied in rule_results.items() if satisfied}

        # Step 3: Calculate Foundation Constraint (G)
        foundation_g = self.constraint_calculator.calculate_foundation_constraint(satisfied_rules)

        # Step 4: Calculate Specialist Prescription (k)
        specialist_k = self.constraint_calculator.calculate_specialist_prescription(satisfied_rules)

        # Step 5: Construct Unified Ranking Vector Q(s, G, k)
        ranking_vector = [safety_floor, foundation_g, specialist_k]

        # Clinical Flag (Predicate Logic)
        is_safe = safety_floor >= 2  # Safe: s ≥ 2 (Minor or No Interaction)
        clinical_flag = "Safe" if is_safe else "Flagged"

        # CPG Compliance Flag
        cpg_compliant = foundation_g == 1

        return {
            'pathway': ' + '.join(drugs),
            'drugs': drugs,
            'n_drugs': len(drugs),
            'n_interactions': len(interactions),
            'safety_floor_s': safety_floor,
            'foundation_g': foundation_g,
            'specialist_k': specialist_k,
            'ranking_vector': ranking_vector,
            'clinical_flag': clinical_flag,
            'cpg_compliant': cpg_compliant,
            'satisfied_rules': sorted(list(satisfied_rules)),
            'rule_details': rule_results,
            'interactions': interactions
        }

    @staticmethod
    def lexicographic_compare(pathway_a: Dict, pathway_b: Dict) -> int:
        """
        Lexicographic comparison of two pathways

        Returns:
        1 if pathway_a > pathway_b (A is better)
        -1 if pathway_a < pathway_b (B is better)
        0 if equal

        Priority: s > G > k
        """
        vec_a = pathway_a['ranking_vector']
        vec_b = pathway_b['ranking_vector']

        for i in range(3):
            if vec_a[i] > vec_b[i]:
                return 1
            elif vec_a[i] < vec_b[i]:
                return -1

        return 0  # Equal

    def rank_pathways(self, pathways_results: List[Dict]) -> List[Dict]:
        """
        Rank multiple pathways using lexicographic ordering
        """
        from functools import cmp_to_key

        # Sort in descending order (best first)
        ranked = sorted(
            pathways_results,
            key=cmp_to_key(lambda a, b: -self.lexicographic_compare(a, b))
        )

        # Add rank numbers
        for i, pathway in enumerate(ranked, 1):
            pathway['rank'] = i

        return ranked

# ============================================================================
# PATHWAY SIMULATOR: DEMONSTRATES THE COMPLETE MATHEMATICAL FRAMEWORK
# ============================================================================

class PathwaySimulator:
    """
    Demonstrates the complete mathematical framework:
    - Four components (Interaction Complexity, Safety Floor, Foundation Constraint, Specialist Score)
    - Final Result (Unified Ranking Vector Q(s, G, k))
    """

    def __init__(self, ddi_dataset_path: str):
        self.ddi_calculator = DDISeverityCalculator(ddi_dataset_path)
        self.ranker = PathwayRanker(self.ddi_calculator)

    def print_section_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    def generate_clinical_recommendations(self, pathways_results: List[Dict]):
        """Generate clinical recommendations grouped by pathway type and severity"""

        # Filter only CPG-compliant pathways (G=1)
        cpg_compliant = [p for p in pathways_results if p['cpg_compliant']]

        if not cpg_compliant:
            print("⚠️  No CPG-compliant pathways found. All pathways must include ACEI/ARB + CCB.")
            return

        # Group pathways by type
        pathway_groups = {
            'foundation': [],  # ACEI/ARB + CCB only
            'foundation_diuretic': [],  # ACEI/ARB + CCB + Diuretic
            'foundation_bb': [],  # ACEI/ARB + CCB + Beta-Blocker
            'foundation_both': []  # ACEI/ARB + CCB + Diuretic + Beta-Blocker
        }

        for pathway in cpg_compliant:
            drug_classes = set(DRUG_TO_CLASS.get(drug) for drug in pathway['drugs'])
            has_diuretic = 'Diuretic' in drug_classes
            has_bb = 'Beta-Blocker' in drug_classes

            if has_diuretic and has_bb:
                pathway_groups['foundation_both'].append(pathway)
            elif has_diuretic:
                pathway_groups['foundation_diuretic'].append(pathway)
            elif has_bb:
                pathway_groups['foundation_bb'].append(pathway)
            else:
                pathway_groups['foundation'].append(pathway)

        # Print grouped recommendations
        self.print_section_header("CLINICAL RECOMMENDATIONS (CPG-Compliant Pathways Only)")
        print("💡 Recommendations will be shown one group at a time.\n")

        # Foundation only
        if pathway_groups['foundation']:
            self._print_pathway_group("FOUNDATION: RAAS Blocker + CCB", pathway_groups['foundation'])
            input("\n⏎ Press ENTER to see next group...")

        # Foundation + Diuretic
        if pathway_groups['foundation_diuretic']:
            self._print_pathway_group("FOUNDATION + DIURETIC", pathway_groups['foundation_diuretic'])
            input("\n⏎ Press ENTER to see next group...")

        # Foundation + Beta-Blocker
        if pathway_groups['foundation_bb']:
            self._print_pathway_group("FOUNDATION + BETA-BLOCKER", pathway_groups['foundation_bb'])
            input("\n⏎ Press ENTER to see next group...")

        # Foundation + Both
        if pathway_groups['foundation_both']:
            self._print_pathway_group("FOUNDATION + DIURETIC + BETA-BLOCKER", pathway_groups['foundation_both'])
            print("\n✅ All recommendations shown.")

    def _print_pathway_group(self, title: str, pathways: List[Dict]):
        """Print a group of pathways separated by ACEI/ARB and severity"""
        print(f"\n{'═' * 80}")
        print(f"📋 {title}")
        print(f"{'═' * 80}")

        # Separate ACEI and ARB pathways
        acei_pathways = []
        arb_pathways = []

        for pathway in pathways:
            drug_classes = [DRUG_TO_CLASS.get(drug) for drug in pathway['drugs']]
            if 'ACEI' in drug_classes:
                acei_pathways.append(pathway)
            elif 'ARB' in drug_classes:
                arb_pathways.append(pathway)

        # Print ACEI combinations
        if acei_pathways:
            print(f"\n🔷 ACEI-Based Combinations:")
            self._print_by_severity(acei_pathways)

        # Print ARB combinations
        if arb_pathways:
            print(f"\n🔶 ARB-Based Combinations:")
            self._print_by_severity(arb_pathways)

    def _print_by_severity(self, pathways: List[Dict]):
        """Print pathways grouped by severity level"""

        # Group by safety floor (severity)
        severity_groups = {
            3: [],  # No Interaction
            2: [],  # Minor
            1: [],  # Moderate
            0: []   # Major
        }

        for pathway in pathways:
            severity_groups[pathway['safety_floor_s']].append(pathway)

        severity_labels = {
            3: "No Interaction",
            2: "Minor",
            1: "Moderate",
            0: "Major"
        }

        # Print each severity group
        for severity in [3, 2, 1, 0]:
            if severity_groups[severity]:
                emoji = "✅" if severity >= 2 else "⚠️" if severity == 1 else "❌"
                print(f"\n   {emoji} {severity_labels[severity]}:")

                for pathway in severity_groups[severity]:
                    print(f"   • {pathway['pathway']}")
                    print(f"     Q(s,G,k) = {pathway['ranking_vector']}")

                    # Show all pairwise interactions
                    for interaction in pathway['interactions']:
                        severity_emoji = "🟢" if interaction['severity_score'] >= 2 else "🟡" if interaction['severity_score'] == 1 else "🔴"
                        print(f"       {severity_emoji} {interaction['pair']}: {interaction['severity_label']}")

                    # Show satisfied rules
                    rules_str = ', '.join(pathway['satisfied_rules'])
                    print(f"     XAI Rules: {{{rules_str}}}")
                    print()

    def print_pathway_result(self, result: Dict):
        """Print detailed pathway evaluation result"""
        print(f"Pathway: {result['pathway']}")
        print(f"Number of drugs (n): {result['n_drugs']}")
        print(f"\n{'─' * 80}")

        # Component 1: Interaction Complexity
        print(f"\n1️⃣  INTERACTION COMPLEXITY")
        print(f"   Using the Combination Formula: C(n, 2) = n! / (2!(n-2)!)")
        print(f"   For n = {result['n_drugs']} drugs:")
        print(f"   C({result['n_drugs']}, 2) = {result['n_interactions']} interaction pair{'s' if result['n_interactions'] != 1 else ''}")

        # Component 2: Safety Floor
        print(f"\n2️⃣  SAFETY FLOOR (s)")
        print(f"   The minimum severity score across all pairs (weakest link principle)")
        print(f"   Severity ranges: 0=Major DDI, 1=Moderate DDI, 2=Minor DDI, 3=No Interaction")
        print(f"\n   Interaction Details:")
        for interaction in result['interactions']:
            severity_emoji = "🟢" if interaction['severity_score'] >= 2 else "🟡" if interaction['severity_score'] == 1 else "🔴"
            print(f"   {severity_emoji} {interaction['pair']}: {interaction['severity_label']} (score={interaction['severity_score']})")
        print(f"\n   Result: s = {result['safety_floor_s']} ({result['clinical_flag']})")

        # Component 3: Foundation Constraint
        print(f"\n3️⃣  FOUNDATION CONSTRAINT (G)")
        print(f"   Indicator Function: G = 1 if CPG Rules A, B, and C are ALL satisfied, 0 otherwise")
        print(f"   This ensures patients receive a RAS blocker plus a Calcium Channel Blocker")
        print(f"\n   XAI Rule Evaluation:")
        for rule, satisfied in result['rule_details'].items():
            status = "✓ Satisfied" if satisfied else "✗ Not Satisfied"
            print(f"   • Rule {rule}: {status}")
        print(f"\n   Foundation Set {{A, B, C}} ⊆ SatisfiedRules? {'Yes' if result['cpg_compliant'] else 'No'}")
        print(f"   Result: G = {result['foundation_g']} ({'CPG-Compliant' if result['cpg_compliant'] else 'Not CPG-Compliant'})")

        # Component 4: Specialist Score
        print(f"\n4️⃣  SPECIALIST SCORE (k)")
        print(f"   Counts how many escalation rules (Diuretics or Beta-blockers) are satisfied")
        print(f"   For resistant hypertension: Escalation Set ℰ = {{D, E}}")
        satisfied_escalation = set(result['satisfied_rules']).intersection(ESCALATION_SET)
        print(f"   Satisfied Escalation Rules: {{{', '.join(satisfied_escalation) if satisfied_escalation else 'None'}}}")
        print(f"   Result: k = {result['specialist_k']}")

        # Final Result: Unified Ranking Vector
        print(f"\n{'═' * 80}")
        print(f"FINAL RESULT: Unified Ranking Vector Q(s, G, k)")
        print(f"{'═' * 80}")
        print(f"   Q(s, G, k) = [{result['safety_floor_s']}, {result['foundation_g']}, {result['specialist_k']}]ᵀ")
        print(f"\n   Lexicographic Ordering: Safety (s) dominates, then CPG compliance (G), then")
        print(f"   specialist prescription (k). Safety dominates because a guideline-compliant")
        print(f"   prescription is medically invalid if it causes a life-threatening drug interaction.")
        print(f"\n{'─' * 80}")

        # Clinical recommendation for non-CPG-compliant pathways
        if result['foundation_g'] == 0:
            self._suggest_cpg_alternatives(result)

        print()

    def _suggest_cpg_alternatives(self, result: Dict):
        """Suggest CPG-compliant alternatives when G=0"""
        print(f"\n⚠️  CLINICAL RECOMMENDATION:")
        print(f"   This pathway is NOT CPG-Compliant (missing ACEI/ARB + CCB foundation).")
        print(f"   International guidelines recommend ACEI/ARB + CCB as first-line therapy.")
        print(f"\n   💡 Suggested CPG-Compliant Alternatives:")

        # Determine what's missing
        drugs = result['drugs']
        drug_classes = set(DRUG_TO_CLASS.get(drug) for drug in drugs)

        has_raas = 'ACEI' in drug_classes or 'ARB' in drug_classes
        has_ccb = 'CCB' in drug_classes

        # Get RAAS blocker or CCB from current pathway
        current_raas = None
        current_ccb = None
        other_drugs = []

        for drug in drugs:
            drug_class = DRUG_TO_CLASS.get(drug)
            if drug_class in ['ACEI', 'ARB']:
                current_raas = drug
            elif drug_class == 'CCB':
                current_ccb = drug
            else:
                other_drugs.append(drug)

        # Generate alternatives
        alternatives = []

        if has_raas and not has_ccb:
            # Has RAAS blocker, missing CCB - suggest adding CCB
            ccb_options = ['Amlodipine', 'Nifedipine', 'Diltiazem']
            for ccb in ccb_options:
                if ccb in DRUG_TO_CLASS:  # Check if in dataset
                    alt_drugs = [current_raas, ccb] + other_drugs
                    alt_result = self.ranker.evaluate_pathway(alt_drugs)
                    if alt_result['cpg_compliant'] and alt_result['safety_floor_s'] >= 2:
                        alternatives.append(alt_result)

        elif has_ccb and not has_raas:
            # Has CCB, missing RAAS blocker - suggest adding ACEI/ARB
            raas_options = ['Ramipril', 'Enalapril', 'Lisinopril', 'Losartan', 'Telmisartan']
            for raas in raas_options:
                if raas in DRUG_TO_CLASS:  # Check if in dataset
                    alt_drugs = [raas, current_ccb] + other_drugs
                    alt_result = self.ranker.evaluate_pathway(alt_drugs)
                    if alt_result['cpg_compliant'] and alt_result['safety_floor_s'] >= 2:
                        alternatives.append(alt_result)

        else:
            # Missing both - suggest common combinations
            combinations = [
                ['Ramipril', 'Amlodipine'],
                ['Enalapril', 'Amlodipine'],
                ['Lisinopril', 'Amlodipine'],
                ['Losartan', 'Amlodipine'],
            ]
            for combo in combinations:
                if all(drug in DRUG_TO_CLASS for drug in combo):
                    alt_drugs = combo + other_drugs
                    alt_result = self.ranker.evaluate_pathway(alt_drugs)
                    if alt_result['cpg_compliant'] and alt_result['safety_floor_s'] >= 2:
                        alternatives.append(alt_result)

        # Sort alternatives by ranking vector
        alternatives = sorted(alternatives, key=lambda x: x['ranking_vector'], reverse=True)

        # Display top 3 alternatives
        if alternatives:
            for i, alt in enumerate(alternatives[:3], 1):
                severity_label = {3: "No Interaction", 2: "Minor", 1: "Moderate", 0: "Major"}[alt['safety_floor_s']]
                print(f"   {i}. {alt['pathway']}: {severity_label} Q{alt['ranking_vector']}")
        else:
            print(f"   (No safe CPG-compliant alternatives found with current drugs)")

    def demonstrate_use_case(self, case_name: str, drugs: List[str], description: str, wait_for_user: bool = True):
        """Demonstrate a single use case"""
        print(f"\n{'▼' * 40}")
        print(f"USE CASE: {case_name}")
        print(f"Description: {description}")
        print(f"{'▼' * 40}\n")

        result = self.ranker.evaluate_pathway(drugs)
        self.print_pathway_result(result)

        if wait_for_user:
            input("\n⏎ Press ENTER to continue to next use case...")

        return result

    def run_exhaustive_simulation(self):
        """
        Run comprehensive simulation with exhaustive use cases
        """
        self.print_section_header("MATHEMATICAL FRAMEWORK DEMONSTRATION")
        print("This simulation demonstrates the complete mathematical system for")
        print("evaluating and ranking hypertension medication pathways.\n")

        all_results = []

        # ====================================================================
        # CATEGORY 1: 2-DRUG COMBINATIONS (n=2)
        # ====================================================================
        self.print_section_header("CATEGORY 1: TWO-DRUG COMBINATIONS (n=2)")
        print("Interaction Pairs: C(2,2) = 1 pair")

        # Use Case 1.1: ACEI + CCB (Foundation Rules A, B, C) - BEST PATHWAY (RANK 1)
        result = self.demonstrate_use_case(
            "1.1 - ACEI + CCB (Optimal Foundation) - RANK 1",
            ['Ramipril', 'Amlodipine'],
            "Tests Foundation Set {A, B, C} satisfaction via RAAS blocker + CCB combo"
        )
        all_results.append(result)

        # Use Case 1.2: Dual RAAS Blockade (ACEI + ARB) - WORST PATHWAY (RANK 14)
        result = self.demonstrate_use_case(
            "1.2 - Dual RAAS Blockade (ACEI + ARB) - RANK 14",
            ['Ramipril', 'Losartan'],
            "Tests contraindicated dual RAAS blockade (expected Moderate DDI, s=1)"
        )
        all_results.append(result)

        # Use Case 1.3: ARB + CCB (Foundation Rules A, B, C)
        result = self.demonstrate_use_case(
            "1.3 - ARB + CCB (Alternative Foundation)",
            ['Losartan', 'Amlodipine'],
            "Tests Foundation Set with ARB instead of ACEI"
        )
        all_results.append(result)

        # Use Case 1.4: ACEI + Thiazide (Rules A, B only - Missing C)
        result = self.demonstrate_use_case(
            "1.4 - ACEI + Thiazide (Incomplete Foundation)",
            ['Enalapril', 'Hydrochlorothiazide'],
            "Tests pathway missing Rule C (no CCB), G=0 expected"
        )
        all_results.append(result)

        # Use Case 1.5: CCB + Beta-Blocker (Missing A, B)
        result = self.demonstrate_use_case(
            "1.5 - CCB + Beta-Blocker (No RAAS Blocker)",
            ['Amlodipine', 'Metoprolol'],
            "Tests pathway without ACEI/ARB, missing Foundation Rules A & B"
        )
        all_results.append(result)

        # ====================================================================
        # CATEGORY 2: 3-DRUG COMBINATIONS (n=3)
        # ====================================================================
        self.print_section_header("CATEGORY 2: THREE-DRUG COMBINATIONS (n=3)")
        print("Interaction Pairs: C(3,2) = 3 pairs")

        # Use Case 2.1: ACEI + CCB + Thiazide (Foundation + No Specialist)
        result = self.demonstrate_use_case(
            "2.1 - ACEI + CCB + Thiazide (Foundation Only)",
            ['Ramipril', 'Amlodipine', 'Hydrochlorothiazide'],
            "Tests Foundation {A,B,C} with HCTZ (no Rule D), k=0"
        )
        all_results.append(result)

        # Use Case 2.2: ACEI + CCB + Indapamide (Foundation + Rule D)
        result = self.demonstrate_use_case(
            "2.2 - ACEI + CCB + Indapamide (Foundation + Specialist D)",
            ['Lisinopril', 'Amlodipine', 'Indapamide'],
            "Tests Foundation {A,B,C} + Escalation Rule D, k=1"
        )
        all_results.append(result)

        # Use Case 2.3: ACEI + CCB + Beta-Blocker (Foundation + Rule E)
        result = self.demonstrate_use_case(
            "2.3 - ACEI + CCB + Beta-Blocker (Foundation + Specialist E)",
            ['Perindopril', 'Nifedipine', 'Bisoprolol'],
            "Tests Foundation {A,B,C} + Escalation Rule E, k=1"
        )
        all_results.append(result)

        # Use Case 2.4: ARB + CCB + Beta-Blocker (Foundation + Rule E)
        result = self.demonstrate_use_case(
            "2.4 - ARB + CCB + Beta-Blocker (ARB-based Foundation + E)",
            ['Telmisartan', 'Amlodipine', 'Metoprolol'],
            "Tests Foundation with ARB + Rule E for high HR phenotype"
        )
        all_results.append(result)

        # ====================================================================
        # CATEGORY 3: 4-DRUG COMBINATIONS (n=4, Maximum)
        # ====================================================================
        self.print_section_header("CATEGORY 3: FOUR-DRUG COMBINATIONS (n=4, MAXIMUM)")
        print("Interaction Pairs: C(4,2) = 6 pairs")

        # Use Case 3.1: ACEI + CCB + Indapamide + Beta-Blocker (Full Stack)
        result = self.demonstrate_use_case(
            "3.1 - ACEI + CCB + Indapamide + Beta-Blocker (Maximum Rules)",
            ['Ramipril', 'Amlodipine', 'Indapamide', 'Bisoprolol'],
            "Tests Foundation {A,B,C} + Both Escalation Rules {D,E}, k=2"
        )
        all_results.append(result)

        # Use Case 3.2: ARB + CCB + Indapamide + Beta-Blocker (ARB Maximum)
        result = self.demonstrate_use_case(
            "3.2 - ARB + CCB + Indapamide + Beta-Blocker (ARB Maximum)",
            ['Losartan', 'Nifedipine', 'Indapamide', 'Nebivolol'],
            "Tests ARB-based Foundation + Both Escalation Rules {D,E}, k=2"
        )
        all_results.append(result)

        # Use Case 3.3: ACEI + CCB + HCTZ + Beta-Blocker (Foundation + E only)
        result = self.demonstrate_use_case(
            "3.3 - ACEI + CCB + HCTZ + Beta-Blocker (Suboptimal Diuretic)",
            ['Enalapril', 'Amlodipine', 'Hydrochlorothiazide', 'Atenolol'],
            "Tests Foundation + Rule E only (HCTZ instead of Indapamide), k=1"
        )
        all_results.append(result)

        # ====================================================================
        # CATEGORY 4: EDGE CASES & SPECIAL SCENARIOS
        # ====================================================================
        self.print_section_header("CATEGORY 4: EDGE CASES & SPECIAL SCENARIOS")

        # Use Case 4.1: Beta-Blocker + CCB (Rate-Control Interaction)
        result = self.demonstrate_use_case(
            "4.1 - Non-Dihydropyridine CCB + Beta-Blocker",
            ['Diltiazem', 'Metoprolol'],
            "Tests potential bradycardia risk (heart rate too low)"
        )
        all_results.append(result)

        # Use Case 4.2: Triple Therapy without Beta-Blocker
        result = self.demonstrate_use_case(
            "4.2 - ACEI + CCB + Indapamide (No Beta-Blocker)",
            ['Ramipril', 'Amlodipine', 'Indapamide'],
            "Tests Foundation + Rule D only, k=1"
        )
        all_results.append(result)

        # ====================================================================
        # COMPARATIVE RANKING DEMONSTRATION
        # ====================================================================
        self.print_section_header("LEXICOGRAPHIC RANKING COMPARISON")

        print("Ranking all evaluated pathways using lexicographic ordering:")
        print("Priority: Safety Floor (s) > CPG Compliance (G) > Specialist Rules (k)\n")

        ranked_pathways = self.ranker.rank_pathways(all_results)

        print(f"{'Rank':<6} {'Pathway':<50} {'Q(s,G,k)':<15} {'Flag':<10} {'CPG':<12}")
        print("─" * 100)

        for pathway in ranked_pathways:
            vector_str = str(pathway['ranking_vector'])
            cpg_str = "✓ Compliant" if pathway['cpg_compliant'] else "✗ Non-Compliant"
            print(f"{pathway['rank']:<6} {pathway['pathway']:<50} {vector_str:<15} {pathway['clinical_flag']:<10} {cpg_str:<12}")

        # ====================================================================
        # DETAILED TOP 3 ANALYSIS
        # ====================================================================
        self.print_section_header("TOP 3 RECOMMENDED PATHWAYS - DETAILED ANALYSIS")

        for i in range(min(3, len(ranked_pathways))):
            pathway = ranked_pathways[i]
            print(f"\n{'━' * 80}")
            print(f"RANK #{pathway['rank']}: {pathway['pathway']}")
            print(f"{'━' * 80}")
            print(f"Ranking Vector Q(s,G,k) = {pathway['ranking_vector']}")
            print(f"Safety Floor (s): {pathway['safety_floor_s']} - {pathway['clinical_flag']}")
            print(f"CPG Compliance (G): {pathway['foundation_g']} - {'CPG-Compliant' if pathway['cpg_compliant'] else 'Not CPG-Compliant'}")
            print(f"Specialist Rules (k): {pathway['specialist_k']}")
            print(f"Satisfied XAI Rules: {{{', '.join(pathway['satisfied_rules'])}}}")

            print(f"\nClinical Rationale:")
            if pathway['cpg_compliant']:
                print(f"  ✓ Meets all Foundation requirements (RAAS blocker + CCB combination)")
            if pathway['specialist_k'] == 2:
                print(f"  ✓ Includes both Indapamide (superior diuretic) and Beta-Blocker (HR control)")
            elif pathway['specialist_k'] == 1:
                if 'D' in pathway['satisfied_rules']:
                    print(f"  ✓ Includes Indapamide for superior CV outcomes")
                if 'E' in pathway['satisfied_rules']:
                    print(f"  ✓ Includes Beta-Blocker for high HR phenotype")

            print(f"\nDrug Interaction Safety:")
            for interaction in pathway['interactions']:
                severity_emoji = "🟢" if interaction['severity_score'] >= 2 else "🔴" if interaction['severity_score'] == 0 else "🟡"
                print(f"  {severity_emoji} {interaction['pair']}: {interaction['severity_label']}")

        # ====================================================================
        # MATHEMATICAL SUMMARY
        # ====================================================================
        self.print_section_header("MATHEMATICAL FRAMEWORK SUMMARY")

        print("Four Mathematical Components:\n")

        print("1. INTERACTION COMPLEXITY:")
        print("   Using the Combination Formula: C(n, 2) = n! / (2!(n-2)!)")
        print("   2 drugs → 1 interaction pair, 3 drugs → 3 pairs, 4 drugs → 6 pairs\n")

        print("2. SAFETY FLOOR (s):")
        print("   The minimum severity score across all pairs (weakest link principle)")
        print("   Severity ranges: 0=Major DDI, 1=Moderate DDI, 2=Minor DDI, 3=No Interaction\n")

        print("3. FOUNDATION CONSTRAINT (G):")
        print("   Indicator Function: G = 1 if CPG Rules A, B, and C are ALL satisfied, 0 otherwise")
        print("   This ensures patients receive a RAS blocker plus a Calcium Channel Blocker\n")

        print("4. SPECIALIST SCORE (k):")
        print("   Counts how many escalation rules (Diuretics or Beta-blockers) are satisfied")
        print("   For resistant hypertension: Escalation Set ℰ = {D, E}\n")

        print("=" * 80)
        print("FINAL RESULT: Unified Ranking Vector Q(s, G, k)")
        print("=" * 80)
        print("   Q(s, G, k) = [s, G, k]ᵀ")
        print("   Lexicographic Ordering: Safety (s) dominates, then CPG compliance (G), then")
        print("   specialist prescription (k). Safety dominates because a guideline-compliant")
        print("   prescription is medically invalid if it causes a life-threatening drug interaction.\n")

        # ====================================================================
        # CLINICAL RECOMMENDATIONS
        # ====================================================================
        self.generate_clinical_recommendations(all_results)

        # ====================================================================
        # STATISTICS
        # ====================================================================
        self.print_section_header("SIMULATION STATISTICS")

        total_pathways = len(all_results)
        safe_pathways = sum(1 for r in all_results if r['clinical_flag'] == 'Safe')
        cpg_compliant = sum(1 for r in all_results if r['cpg_compliant'])
        with_specialist = sum(1 for r in all_results if r['specialist_k'] > 0)

        print(f"Total Pathways Evaluated: {total_pathways}")
        print(f"Safe Pathways (s ≥ 2): {safe_pathways} ({safe_pathways/total_pathways*100:.1f}%)")
        print(f"CPG-Compliant (G = 1): {cpg_compliant} ({cpg_compliant/total_pathways*100:.1f}%)")
        print(f"With Specialist Rules (k > 0): {with_specialist} ({with_specialist/total_pathways*100:.1f}%)")

        print(f"\nPathway Complexity Distribution:")
        for n in [2, 3, 4]:
            count = sum(1 for r in all_results if r['n_drugs'] == n)
            max_interactions = n * (n-1) // 2
            print(f"  {n}-drug pathways: {count} (C({n},2) = {max_interactions} interactions each)")

        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)

        return ranked_pathways

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  HYPERTENSION DDI DETECTION - MATHEMATICAL PATHWAY SIMULATOR".center(78) + "█")
    print("█" + "  Demonstrating Knowledge-Driven Explainability Framework".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")

    # Initialize simulator with the DDI dataset
    simulator = PathwaySimulator('FYP_DrugBank_Inclusive.csv')

    # Run exhaustive simulation
    ranked_results = simulator.run_exhaustive_simulation()

    # Export results for further analysis
    print("\nExporting results to JSON...")
    with open('pathway_simulation_results.json', 'w') as f:
        json.dump(ranked_results, f, indent=2)
    print("✓ Results saved to: pathway_simulation_results.json")

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  Thank you for reviewing this mathematical demonstration!".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
