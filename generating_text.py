import os
import time
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)

MODEL_NAME = "models/gemini-3.5-flash-lite"


# ============================================================
# 3. FILES
# ============================================================

INPUT_FILE = "data/fraud_oracle.csv"
OUTPUT_FILE = "data/fraud_oracle_with_text.csv"


# ============================================================
# 4. LOAD DATA
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print("Existing output file found.")
    print("Resuming from previous progress...")

    df = pd.read_csv(OUTPUT_FILE)

else:

    print("Creating new output file...")

    df = pd.read_csv(INPUT_FILE)

    df["ClaimNarrative"] = ""


print("\nDataset shape:", df.shape)


# ============================================================
# 5. COLUMNS USED FOR NARRATIVE
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
# 6. BUILD PROMPT
# ============================================================

def build_prompt(row):

    claim_details = "\n".join(
        f"- {field}: {row[field]}"
        for field in NARRATIVE_FIELDS
    )

    return f"""
You are generating a SYNTHETIC vehicle insurance claim
narrative for an academic machine-learning project.

Use ONLY the information provided below.

Write a realistic 2 to 4 sentence insurance claim narrative.

Rules:
- Do NOT mention fraud.
- Do NOT mention fraudulent behavior.
- Do NOT mention fraud probability.
- Do NOT mention machine learning.
- Do NOT mention classification.
- Do NOT say that the claimant is suspicious.
- Do NOT reveal or infer the target label.
- Do NOT invent injuries, witnesses, police statements,
  damage details, locations, or events that are not provided.
- Keep the narrative neutral and factual.
- Use natural insurance claim language.
- Do not use bullet points.
- Do not include a heading.
- Output ONLY the narrative.

Claim information:

{claim_details}
"""


# ============================================================
# 7. GENERATE ONE NARRATIVE
# ============================================================

def generate_narrative(row, max_retries=5):

    prompt = build_prompt(row)

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=150
                )
            )

            text = response.text.strip()

            if text:
                return text

            raise ValueError(
                "Empty response from Gemini."
            )

        except Exception as e:

            print(
                f"\nError: {e}"
            )

            if attempt < max_retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "Maximum retries reached."
                )

    return ""


# ============================================================
# 8. BATCH SETTINGS
# ============================================================

BATCH_SIZE = 50

total_rows = len(df)


# ============================================================
# 9. FIND ALREADY COMPLETED ROWS
# ============================================================

completed = (
    df["ClaimNarrative"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

completed_count = completed.sum()

print(
    f"\nAlready completed: "
    f"{completed_count}/{total_rows}"
)

print(
    f"Remaining: "
    f"{total_rows - completed_count}"
)


# ============================================================
# 10. PROCESS IN BATCHES OF 50
# ============================================================

for start in range(
    0,
    total_rows,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_rows
    )

    # Check whether this batch is already complete
    batch_completed = (
        df.loc[start:end - 1, "ClaimNarrative"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    if batch_completed:

        print(
            f"\nBatch {start + 1}-{end} "
            f"already completed. Skipping..."
        )

        continue


    print(
        "\n" + "=" * 70
    )

    print(
        f"Processing batch: "
        f"{start + 1} → {end}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Generate 50 rows
    # --------------------------------------------------------

    for idx in tqdm(
        range(start, end),
        desc=f"Rows {start + 1}-{end}"
    ):

        # Skip if already generated
        existing_text = df.at[
            idx,
            "ClaimNarrative"
        ]

        if (
            pd.notna(existing_text)
            and str(existing_text).strip()
        ):
            continue


        narrative = generate_narrative(
            df.iloc[idx]
        )


        df.at[
            idx,
            "ClaimNarrative"
        ] = narrative


        # Small delay between requests
        time.sleep(0.2)


    # --------------------------------------------------------
    # SAVE AFTER EVERY 50 ROWS
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        f"\n✅ Saved batch {start + 1}-{end}"
    )

    print(
        f"File: {OUTPUT_FILE}"
    )

    # Optional pause between batches
    time.sleep(2)


# ============================================================
# 11. FINAL CHECK
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

completed = (
    df["ClaimNarrative"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

print("\n" + "=" * 70)
print("GENERATION FINISHED")
print("=" * 70)

print(
    f"Completed narratives: "
    f"{completed.sum()}/{len(df)}"
)

print(
    "Remaining:",
    len(df) - completed.sum()
)

print(
    "\nOutput:",
    OUTPUT_FILE
)