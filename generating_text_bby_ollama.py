import os
import time
import pandas as pd
from tqdm import tqdm
import ollama


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/fraud_oracle.csv"
OUTPUT_FILE = "data/fraud_oracle_with_text_by_ollama.csv"

MODEL_NAME = "mistral"

BATCH_SIZE = 50


# ============================================================
# LOAD DATA
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print("Existing output file found.")
    print("Resuming previous progress...")

    df = pd.read_csv(OUTPUT_FILE)

else:

    print("Creating new output file...")

    df = pd.read_csv(INPUT_FILE)

    df["ClaimNarrative"] = ""


print("\nDataset shape:", df.shape)


# ============================================================
# COLUMNS USED TO CREATE THE NARRATIVE
# ============================================================

NARRATIVE_FIELDS = [
    "Month",
    "WeekOfMonth",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "WeekOfMonthClaimed",
    "Fault",
    "VehicleCategory",
    "VehiclePrice",
    "BasePolicy",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgeOfVehicle",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "NumberOfSuppliments"
]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col
    for col in NARRATIVE_FIELDS
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(row):

    claim_details = "\n".join(
        f"{field}: {row[field]}"
        for field in NARRATIVE_FIELDS
    )

    prompt = f"""
You are generating a SYNTHETIC vehicle insurance claim
narrative for an academic machine learning project.

Create a realistic 2 to 4 sentence insurance claim narrative
using ONLY the structured claim information provided below.

STRICT RULES:

1. Do not mention fraud.
2. Do not mention fraudulent behavior.
3. Do not mention fraud probability.
4. Do not mention machine learning.
5. Do not mention classification.
6. Do not say the claimant is suspicious.
7. Do not reveal a target label.
8. Do not invent injuries.
9. Do not invent witnesses.
10. Do not invent police statements.
11. Do not invent damage details that are not provided.
12. Do not invent an accident location.
13. Keep the narrative neutral and factual.
14. Use natural insurance-claim language.
15. Do not use bullet points.
16. Do not include a heading.
17. Output ONLY the narrative.

Structured claim information:

{claim_details}
"""

    return prompt


# ============================================================
# GENERATE ONE NARRATIVE
# ============================================================

def generate_narrative(row, max_retries=3):

    prompt = build_prompt(row)

    for attempt in range(max_retries):

        try:

            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.3
                }
            )

            text = response["message"]["content"].strip()

            if text:
                return text

            raise ValueError(
                "Ollama returned empty response."
            )

        except Exception as e:

            print(
                f"\nGeneration error "
                f"(attempt {attempt + 1}): {e}"
            )

            if attempt < max_retries - 1:

                time.sleep(2)

            else:

                return ""

    return ""


# ============================================================
# TEST FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 80)
print("TESTING FIRST 5 ROWS")
print("=" * 80)

for i in range(min(5, len(df))):

    narrative = generate_narrative(
        df.iloc[i]
    )

    print("\nROW:", i)

    print(
        "STRUCTURED:"
    )

    for field in NARRATIVE_FIELDS:
        print(
            f"  {field}: {df.iloc[i][field]}"
        )

    print(
        "\nGENERATED:"
    )

    print(narrative)

    print("-" * 80)


# ============================================================
# ASK BEFORE FULL GENERATION
# ============================================================

choice = input(
    "\nGenerate the remaining narratives? (yes/no): "
).strip().lower()

if choice != "yes":

    print("\nStopped after test.")
    exit()


# ============================================================
# ENSURE NARRATIVE COLUMN EXISTS
# ============================================================

if "ClaimNarrative" not in df.columns:

    df["ClaimNarrative"] = ""


# ============================================================
# COUNT COMPLETED ROWS
# ============================================================

completed_mask = (
    df["ClaimNarrative"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

completed_count = completed_mask.sum()

print(
    f"\nCompleted: "
    f"{completed_count}/{len(df)}"
)

print(
    f"Remaining: "
    f"{len(df) - completed_count}"
)


# ============================================================
# PROCESS IN BATCHES OF 50
# ============================================================

for start in range(
    0,
    len(df),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(df)
    )

    # ----------------------------------------------
    # Check whether entire batch is already complete
    # ----------------------------------------------

    batch = df.iloc[start:end]

    batch_completed = (
        batch["ClaimNarrative"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    if batch_completed:

        print(
            f"\nSkipping batch "
            f"{start + 1}-{end} "
            f"(already completed)"
        )

        continue


    print("\n" + "=" * 80)

    print(
        f"PROCESSING BATCH "
        f"{start + 1} → {end}"
    )

    print("=" * 80)


    # ----------------------------------------------
    # Generate narratives
    # ----------------------------------------------

    for idx in tqdm(
        range(start, end),
        desc=f"Batch {start + 1}-{end}"
    ):

        existing = df.at[
            idx,
            "ClaimNarrative"
        ]

        # Skip if already generated
        if (
            pd.notna(existing)
            and str(existing).strip() != ""
        ):
            continue


        narrative = generate_narrative(
            df.iloc[idx]
        )


        df.at[
            idx,
            "ClaimNarrative"
        ] = narrative


    # ----------------------------------------------
    # SAVE AFTER EVERY 50
    # ----------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✅ Batch {start + 1}-{end} saved."
    )

    print(
        f"File: {OUTPUT_FILE}"
    )


# ============================================================
# FINAL SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

completed_mask = (
    df["ClaimNarrative"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

print(
    "Total rows:",
    len(df)
)

print(
    "Generated narratives:",
    completed_mask.sum()
)

print(
    "Missing narratives:",
    (~completed_mask).sum()
)

print(
    "Output file:",
    OUTPUT_FILE
)