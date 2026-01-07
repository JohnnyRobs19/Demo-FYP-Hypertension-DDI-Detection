import pandas as pd
from itertools import combinations
import argparse

# ==========================================
# COMMAND LINE ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description='Generate drug interaction pairs')
parser.add_argument('--demo', action='store_true',
                    help='Create demo subset from FYP_DrugBank_Inclusive.csv (covers all severities and XAI rules)')
args = parser.parse_args()

# ==========================================
# 1. DEFINE OFFICIAL DRUG LIST (Malaysian CPG / MIMS 2018)
# ==========================================
drugs = {
    "ACEI": ["Captopril", "Enalapril", "Lisinopril", "Perindopril", "Ramipril", "Imidapril"],
    "ARB": ["Candesartan", "Irbesartan", "Losartan", "Telmisartan", "Valsartan", "Olmesartan"],
    "Beta-Blocker": ["Acebutolol", "Atenolol", "Betaxolol", "Bisoprolol", "Metoprolol", "Nebivolol", "Propranolol"],
    # Note: We group them as CCB here, but remember Diltiazem/Verapamil are Non-Dihydro (Heart rate risks)
    "CCB": ["Amlodipine", "Felodipine", "Isradipine", "Lercanidipine", "Nifedipine", "Diltiazem", "Verapamil"],
    "Diuretic": ["Hydrochlorothiazide", "Indapamide", "Amiloride"]
}

# ==========================================
# 2. FLATTEN LIST & GENERATE PAIRS
# ==========================================
all_drugs = []
for category, drug_list in drugs.items():
    # Clean class names (optional, but good for consistency)
    clean_class = "CCB" if "CCB" in category else category
    for drug in drug_list:
        all_drugs.append({"Name": drug, "Class": clean_class})

# ==========================================
# 3. CREATE DATAFRAME
# ==========================================

if args.demo:
    # ==========================================
    # DEMO MODE: Select strategic pairs from existing data
    # ==========================================
    print("="*80)
    print("DEMO MODE: Creating strategic subset from FYP_DrugBank_Inclusive.csv")
    print("="*80)

    # Load existing data
    try:
        full_df = pd.read_csv("FYP_DrugBank_Inclusive.csv")
    except FileNotFoundError:
        print("❌ Error: FYP_DrugBank_Inclusive.csv not found!")
        print("   Please run the scraper first to generate the full dataset.")
        exit(1)

    # Count XAI rules for each row
    def count_xai_rules(row):
        count = 0
        for col in ['XAI_Rule_A_Mortality', 'XAI_Rule_B_Tolerability',
                    'XAI_Rule_C_CCB_RAAS_Combo', 'XAI_Rule_D_Diuretic', 'XAI_Rule_E_BetaBlocker']:
            if pd.notna(row[col]) and str(row[col]).strip() != '':
                count += 1
        return count

    full_df['XAI_Rule_Count'] = full_df.apply(count_xai_rules, axis=1)

    demo_pairs = []

    print("\n🎯 STRATEGIC PAIR SELECTION")
    print("   Goal: 4 pairs × 1 per severity × varied XAI rule combinations")
    print("-"*80)

    # Strategy: Find one pair per severity with different XAI rule counts
    # This demonstrates variety in clinical explanations

    # Pair 1: Major severity with 2 rules (e.g., ACEI+ARB triggers Rules A, B)
    print("\n1. MAJOR severity (2 XAI rules)...")
    major = full_df[(full_df['Final_Severity'] == 'Major') & (full_df['XAI_Rule_Count'] == 2)].head(1)
    if major.empty:
        major = full_df[full_df['Final_Severity'] == 'Major'].head(1)
    if not major.empty:
        demo_pairs.append(major.iloc[0])
        print(f"   {major.iloc[0]['Drug_A_Name']:15s} + {major.iloc[0]['Drug_B_Name']:15s}")
        print(f"   → {major.iloc[0]['XAI_Rule_Count']} rules triggered")

    # Pair 2: Moderate severity with Rule E (Beta-Blocker)
    print("\n2. MODERATE severity (with Rule E - Beta-Blocker)...")
    moderate = full_df[(full_df['Final_Severity'] == 'Moderate') &
                       (full_df['XAI_Rule_E_BetaBlocker'].notna()) &
                       (full_df['XAI_Rule_E_BetaBlocker'].str.strip() != '')].head(1)
    if moderate.empty:
        moderate = full_df[(full_df['Final_Severity'] == 'Moderate') & (full_df['XAI_Rule_Count'] >= 2)].head(1)
    if moderate.empty:
        moderate = full_df[full_df['Final_Severity'] == 'Moderate'].head(1)
    if not moderate.empty:
        demo_pairs.append(moderate.iloc[0])
        print(f"   {moderate.iloc[0]['Drug_A_Name']:15s} + {moderate.iloc[0]['Drug_B_Name']:15s}")
        print(f"   → {moderate.iloc[0]['XAI_Rule_Count']} rules triggered")

    # Pair 3: Minor severity with Rule C (CCB+RAAS - 3 rules: A, B, C)
    print("\n3. MINOR severity (with Rule C - CCB+RAAS combo)...")
    minor = full_df[(full_df['Final_Severity'] == 'Minor') &
                    (full_df['XAI_Rule_C_CCB_RAAS_Combo'].notna()) &
                    (full_df['XAI_Rule_C_CCB_RAAS_Combo'].str.strip() != '')].head(1)
    if minor.empty:
        minor = full_df[full_df['Final_Severity'] == 'Minor'].head(1)
    if not minor.empty:
        demo_pairs.append(minor.iloc[0])
        print(f"   {minor.iloc[0]['Drug_A_Name']:15s} + {minor.iloc[0]['Drug_B_Name']:15s}")
        print(f"   → {minor.iloc[0]['XAI_Rule_Count']} rules triggered")

    # Pair 4: NoInteraction with Rule D (Diuretic - 3 rules: A, B, D)
    print("\n4. NOINTERACTION severity (with Rule D - Diuretic)...")
    nointeraction = full_df[(full_df['Final_Severity'] == 'NoInteraction') &
                            (full_df['XAI_Rule_D_Diuretic'].notna()) &
                            (full_df['XAI_Rule_D_Diuretic'].str.strip() != '')].head(1)
    if nointeraction.empty:
        nointeraction = full_df[full_df['Final_Severity'] == 'NoInteraction'].head(1)
    if not nointeraction.empty:
        demo_pairs.append(nointeraction.iloc[0])
        print(f"   {nointeraction.iloc[0]['Drug_A_Name']:15s} + {nointeraction.iloc[0]['Drug_B_Name']:15s}")
        print(f"   → {nointeraction.iloc[0]['XAI_Rule_Count']} rules triggered")

    # Remove duplicates
    df = pd.DataFrame(demo_pairs).drop_duplicates(subset=['Drug_A_Name', 'Drug_B_Name'])

    # Keep only essential columns for scraping
    df = df[['Drug_A_Name', 'Drug_B_Name', 'Drug_A_Class', 'Drug_B_Class']].copy()

    print(f"\n{'='*80}")
    print(f"✅ Demo Subset Created: {len(df)} pairs")
    print(f"{'='*80}")

    # Save
    output_file = "demo_pairs.csv"
    df.to_csv(output_file, index=False)
    print(f"File saved as: {output_file}")
    print(f"\nNext step: python drugbank_ddi_scraper.py {output_file}")

else:
    # ==========================================
    # NORMAL MODE: Generate all 406 pairs
    # ==========================================
    # Generate all unique pairs.
    # This INCLUDES "Bad" pairs (ACEI+ARB) and "Duplication" pairs (ACEI+ACEI).
    drug_pairs = list(combinations(all_drugs, 2))

    data = []
    for drug_a, drug_b in drug_pairs:
        data.append({
            # --- IDENTITY ---
            "Drug_A_Name": drug_a["Name"],
            "Drug_B_Name": drug_b["Name"],
            "Drug_A_Class": drug_a["Class"],
            "Drug_B_Class": drug_b["Class"],

            # --- VALIDATION DATA (To be Scraped) ---
            "DrugsCom_Severity": "TBD",   # Major/Moderate/Minor/None
            "DrugsCom_Text": "TBD",       # <--- ADDED: Helps you verify conflicts manually
            "DrugBank_Severity": "TBD",   # Major/Moderate/Minor/None
            "DrugBank_Text": "TBD",       # <--- ADDED: Helps you verify conflicts manually

            # --- MODEL TARGETS ---
            "Final_Severity": "TBD",      # The Ground Truth for your ML Model
            "Risk_Score": 0.0             # 0.2 (Major) to 1.0 (None) for Math Model
        })

    df = pd.DataFrame(data)

    # ==========================================
    # 4. VERIFICATION & EXPORT
    # ==========================================
    # Verify "Same Class" pairs exist (e.g. Captopril + Enalapril)
    same_class_count = len(df[df['Drug_A_Class'] == df['Drug_B_Class']])
    acei_arb_count = len(df[((df['Drug_A_Class'] == 'ACEI') & (df['Drug_B_Class'] == 'ARB')) | ((df['Drug_A_Class'] == 'ARB') & (df['Drug_B_Class'] == 'ACEI'))])

    print(f"✅ Template Generated Successfully")
    print(f"Total Pairs: {len(df)}")
    print(f"Duplication Checks (Same Class) Included: {same_class_count}")
    print(f"Major Interaction Checks (ACEI+ARB) Included: {acei_arb_count}")

    # Save the file - THIS is the file your Scraper will read
    df.to_csv("FYP_Drug_Interaction_Template.csv", index=False)
    print("File saved as: FYP_Drug_Interaction_Template.csv")
