"""
Phase 5: Batched RAG run across the test set.

Packages BATCH_SIZE test molecules (each with its OWN top-k retrieved neighbors)
into a single prompt, requesting one JSON array response covering all of them.
This cuts total request count by ~BATCH_SIZE-fold, to stay under daily free-tier caps.

Run `check_quota_reminder()` output and your AI Studio dashboard before relying
on any specific RPD number - free tier limits vary by project/date.
"""

import pandas as pd
import os
import json
import re
import time
import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from google import genai
from google.genai import types
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

INPUT_DIR = r"C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\PhD\AI\AqSolPred"
TRAIN_FEAT = os.path.join(INPUT_DIR, "train_featurized.csv")
TEST_FEAT = os.path.join(INPUT_DIR, "test_featurized.csv")
RESULTS_CSV = os.path.join(INPUT_DIR, "rag_results_batched.csv")

K = 10
MODEL_NAME = "gemini-2.5-flash-lite"
DELAY_BETWEEN_CALLS_SEC = 7         # RPM=10 on flash-lite -> safely under 1 call every 6s
BATCH_SIZE = 20                     # stepped down from 80 to diagnose 503 errors - will tune back up
N_MOLECULES_TO_RUN = 1996
MAX_OUTPUT_TOKENS = 60 * BATCH_SIZE + 300

PILLAR1_COLS = ["fr_ketone", "fr_aldehyde", "fr_ester", "fr_ether", "fr_amide",
                "fr_amine", "fr_aniline", "fr_phenol", "fr_alcohol",
                "fr_carboxylic_acid", "fr_halogen", "fr_sulfonamide",
                "fr_benzene", "fr_pyridine", "fr_nitrile"]
PILLAR2_COLS = ["balaban_j", "bertz_ct", "kappa1", "kappa2",
                "ring_count", "aromatic_ring_count", "rotatable_bonds"]

def bitstring_to_bitvect(bitstring):
    bv = ExplicitBitVect(len(bitstring))
    for i, bit in enumerate(bitstring):
        if bit == "1":
            bv.SetBit(i)
    return bv

def retrieve_top_k(query_bitstring, train_fps, train_df, k=K):
    query_bv = bitstring_to_bitvect(query_bitstring)
    similarities = DataStructs.BulkTanimotoSimilarity(query_bv, train_fps)
    top_k_idx = np.argsort(similarities)[::-1][:k]
    results = train_df.iloc[top_k_idx].copy()
    results["similarity"] = [similarities[i] for i in top_k_idx]
    return results

def descriptor_summary(row):
    nonzero_p1 = {c: int(row[c]) for c in PILLAR1_COLS if row[c] > 0}
    p2 = {c: round(float(row[c]), 2) for c in PILLAR2_COLS}
    return f"functional groups present: {nonzero_p1 if nonzero_p1 else 'none detected'}; topology: {p2}"

def build_batched_prompt(batch_df, train_df, train_fps, k=K):
    """
    Build one prompt covering multiple query molecules, each with its own
    retrieved neighbor block, requesting one JSON array as output.
    """
    instructions = (
        "You are predicting aqueous solubility (LogS, log10 of solubility in mol/L) "
        "for organic molecules. For EACH query molecule below, you are given its own "
        "set of structurally similar reference molecules with known LogS values, ranked "
        "by Tanimoto similarity. Use chemical reasoning about which reference molecules "
        "are most mechanistically relevant (not just most similar) to estimate each "
        "query's LogS independently - do not let one query's reference set influence "
        "another query's prediction.\n\n"
        "Respond with ONLY a JSON array, one object per query molecule, in this exact "
        "format (keep reasoning very brief, max 10 words). The \"id\" field MUST be "
        "copied EXACTLY as given in the query (e.g. \"B-772\", including any letter "
        "prefix and dash - do NOT convert it to a plain number):\n"
        '[{"id": "<string, copied exactly from query>", "logS_pred": <float>, '
        '"confidence": "<low|medium|high>", "reasoning": "<brief phrase>"}, ...]\n\n'
    )

    blocks = []
    max_sims = {}
    for _, query_row in batch_df.iterrows():
        neighbors = retrieve_top_k(query_row["morgan_fp"], train_fps, train_df, k=k)
        max_sims[query_row["ID"]] = neighbors["similarity"].max()

        examples_text = ""
        for _, nrow in neighbors.iterrows():
            examples_text += (
                f"  - SMILES: {nrow['canonical_smiles']} | LogS: {nrow['Solubility']:.3f} | "
                f"sim: {nrow['similarity']:.3f}\n"
            )

        block = (
            f"=== Query id={query_row['ID']} ===\n"
            f"Query SMILES: {query_row['canonical_smiles']} | {descriptor_summary(query_row)}\n"
            f"Reference molecules (most similar first):\n{examples_text}"
        )
        blocks.append(block)

    full_prompt = instructions + "\n".join(blocks)
    return full_prompt, max_sims

def clean_json_response(raw_text):
    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = max(parts, key=len).strip()
        cleaned = cleaned.removeprefix("json").strip()
    start_idx = cleaned.find("[")
    if start_idx > 0:
        cleaned = cleaned[start_idx:]
    end_idx = cleaned.rfind("]")
    if end_idx != -1:
        cleaned = cleaned[:end_idx + 1]
    return cleaned

def regex_fallback_parse_array(raw_text):
    """Last-resort: extract (id, logS_pred) pairs via regex if JSON parsing fails entirely."""
    matches = re.findall(
        r'"id"\s*:\s*"?([\w-]+)"?.*?"logS_pred"\s*:\s*(-?\d+\.?\d*)',
        raw_text, re.DOTALL
    )
    return [{"id": m[0], "logS_pred": float(m[1]), "confidence": "unknown",
              "reasoning": "(parsed via regex fallback)"} for m in matches]

class ServerOverloadedError(Exception):
    """Raised on 503 - signals the caller to stop the whole run, not just this batch."""
    pass

def call_gemini(prompt, client):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            )
        )
        return response.text
    except Exception as e:
        error_str = str(e)
        if "503" in error_str or "UNAVAILABLE" in error_str:
            raise ServerOverloadedError(
                "Server returned 503 (overloaded). Stopping run to preserve daily quota - "
                "progress is checkpointed, safe to re-run later once load clears."
            )
        raise

def load_existing_results():
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=["id", "smiles", "true_logS", "pred_logS",
                                  "confidence", "reasoning", "max_similarity"])

def main():
    client = genai.Client()

    train_df = pd.read_csv(TRAIN_FEAT)
    test_df = pd.read_csv(TEST_FEAT).iloc[:N_MOLECULES_TO_RUN]
    train_fps = [bitstring_to_bitvect(fp) for fp in train_df["morgan_fp"]]

    results_df = load_existing_results()
    already_done_ids = set(results_df["id"].astype(str))

    remaining_df = test_df[~test_df["ID"].astype(str).isin(already_done_ids)]
    print(f"{len(test_df)} total, {len(already_done_ids)} already done, {len(remaining_df)} remaining.")

    batches = [remaining_df.iloc[i:i + BATCH_SIZE] for i in range(0, len(remaining_df), BATCH_SIZE)]
    print(f"Running {len(batches)} batches of up to {BATCH_SIZE} molecules each.")

    if batches:
        sample_prompt, _ = build_batched_prompt(batches[0], train_df, train_fps, k=K)
        approx_tokens = len(sample_prompt) / 4
        print(f"First batch prompt size: {len(sample_prompt):,} characters (~{approx_tokens:,.0f} tokens estimated)")
        print(f"(TPM ceiling is 250,000 - if this number is anywhere close, reduce BATCH_SIZE or K)")

    for batch_num, batch_df in enumerate(batches):
        prompt, max_sims = build_batched_prompt(batch_df, train_df, train_fps, k=K)

        try:
            raw_response = call_gemini(prompt, client)
            cleaned = clean_json_response(raw_response)

            try:
                predictions = json.loads(cleaned)
            except json.JSONDecodeError:
                predictions = regex_fallback_parse_array(raw_response)
                if not predictions:
                    print(f"BATCH {batch_num} PARSE FAILURE. Raw response:\n---\n{raw_response}\n---")
                    time.sleep(DELAY_BETWEEN_CALLS_SEC)
                    continue
                print(f"  (recovered batch {batch_num} via regex fallback)")

            new_rows = []
            unmatched_preds = []
            for pred in predictions:
                pred_id = str(pred.get("id", "")).strip()
                match = batch_df[batch_df["ID"].astype(str).str.strip() == pred_id]
                if match.empty:
                    # fallback 1: model may have stripped a letter prefix (e.g. "772" instead of "B-772")
                    if pred_id.isdigit():
                        match = batch_df[batch_df["ID"].astype(str).str.strip().str.endswith("-" + pred_id)]
                if match.empty:
                    # fallback 2: try matching on SMILES if the model echoed it back
                    pred_smiles = pred.get("smiles", "").strip()
                    if pred_smiles:
                        match = batch_df[batch_df["canonical_smiles"].astype(str).str.strip() == pred_smiles]
                if match.empty:
                    unmatched_preds.append(pred)
                    continue
                true_row = match.iloc[0]
                new_rows.append({
                    "id": true_row["ID"],
                    "smiles": true_row["canonical_smiles"],
                    "true_logS": true_row["Solubility"],
                    "pred_logS": pred["logS_pred"],
                    "confidence": pred.get("confidence", "unknown"),
                    "reasoning": pred.get("reasoning", ""),
                    "max_similarity": max_sims.get(true_row["ID"], None),
                })

            if unmatched_preds and not new_rows:
                print(f"BATCH {batch_num}: JSON parsed ({len(predictions)} items) but ZERO IDs matched.")
                print(f"  Expected IDs in this batch: {list(batch_df['ID'].astype(str))[:5]}... "
                      f"({len(batch_df)} total)")
                print(f"  Sample returned predictions: {predictions[:3]}")
                print(f"  Stopping run so this can be diagnosed before burning more quota.")
                return

            results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
            results_df.to_csv(RESULTS_CSV, index=False)

            print(f"Batch {batch_num + 1}/{len(batches)} done. "
                  f"{len(new_rows)}/{len(batch_df)} molecules parsed successfully. "
                  f"Total so far: {len(results_df)}")

        except ServerOverloadedError as e:
            print(f"\n{e}")
            print(f"Stopped at batch {batch_num}/{len(batches)}. {len(results_df)} molecules saved so far.")
            print("Re-run the script later (e.g. in 30-60 min) to resume from here automatically.")
            return  # hard stop - do not touch remaining batches or print metrics on a partial/interrupted run

        except Exception as e:
            print(f"ERROR (API call failed) on batch {batch_num}: {e}. Will retry remaining on next run.")

        time.sleep(DELAY_BETWEEN_CALLS_SEC)

    if len(results_df) > 1:
        rmse = np.sqrt(mean_squared_error(results_df["true_logS"], results_df["pred_logS"]))
        mae = mean_absolute_error(results_df["true_logS"], results_df["pred_logS"])
        r2 = r2_score(results_df["true_logS"], results_df["pred_logS"])
        print(f"\n=== METRICS over {len(results_df)} molecules ===")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE:  {mae:.3f}")
        print(f"R2:   {r2:.3f}")

if __name__ == "__main__":
    main()
