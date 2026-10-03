"""
Phase 4: batch evaluation on the 1,500-claim synthetic set.

1. Contract check: the agent must reproduce risk_score, confidence, risk_flags and reasoning
   for every row exactly (template mode, so the result is deterministic).
2. Detection metrics against fraud_confirmed, using Pega's governance thresholds
   (Blueprint 10.13: FraudLowThreshold 30, FraudMediumThreshold 60, FraudHighThreshold 80).
3. Shadow challenger: a scikit-learn logistic regression on an out-of-time split. It is
   reported for comparison only and is NEVER used in agent responses (spec s7 and s13: no ML
   model until real SIU-confirmed labels exist).

    python evaluate.py [--csv data/fra_synthetic_claims_1500.csv]
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from fra_agent import FraudRiskAgent, row_to_request

# Request facts only: identifiers, hashes, expected outputs and the two availability flags
# (confidence inputs, not fraud signals) are left out.
NUMERIC = ["vehicle_policy_mismatch", "potential_duplicate", "claims_last_24_months",
           "days_since_policy_start", "late_reported"]


def contract_check(df: pd.DataFrame) -> pd.Series:
    agent = FraudRiskAgent()  # template reasoning: deterministic
    results = [agent.assess(row_to_request(row)) for row in df.to_dict("records")]
    same = [
        r["status"] == "COMPLETED"
        and r["risk_score"] == row.risk_score
        and r["confidence"] == row.confidence
        and r["risk_flags"] == json.loads(row.risk_flags)
        and r["reasoning"] == row.reasoning
        for r, row in zip(results, df.itertuples())
    ]
    print(f"Contract reproduction: {sum(same)}/{len(df)} rows identical "
          "(risk_score, confidence, risk_flags, reasoning)")
    return pd.Series([r.get("risk_score") for r in results], index=df.index)


def detection_metrics(y: pd.Series, score: pd.Series) -> None:
    print(f"\nRule engine vs fraud_confirmed (base rate {y.mean():.1%}, {int(y.sum())} positives)")
    print(f"  PR-AUC {average_precision_score(y, score):.3f}   ROC-AUC {roc_auc_score(y, score):.3f}")
    rows = [(30, "Medium band or above"), (60, "Senior escalation (DT-01)"), (80, "Fraud Investigation referral")]
    for threshold, name in rows:
        flagged = score >= threshold
        print(f"  score >= {threshold:<3} {name:<30} {flagged.mean():6.1%} of claims   "
              f"precision {y[flagged].mean():6.1%}   recall {flagged[y == 1].mean():6.1%}")


def shadow_challenger(df: pd.DataFrame, rule_score: pd.Series) -> None:
    df = df.assign(rule_score=rule_score).sort_values("loss_reported_on")
    cut = int(len(df) * 0.7)  # out-of-time: train on older claims, test on newer ones
    train, test = df.iloc[:cut], df.iloc[cut:]
    features = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cause", OneHotEncoder(handle_unknown="ignore"), ["loss_cause"]),
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True), "loss_description"),
    ])
    model = make_pipeline(features, LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(train, train.fraud_confirmed)
    prob = model.predict_proba(test)[:, 1]

    y = test.fraud_confirmed
    print(f"\nShadow challenger (not used in responses), out-of-time test set: {len(test)} claims "
          f"reported {test.loss_reported_on.min()} to {test.loss_reported_on.max()}")
    print(f"  rules       PR-AUC {average_precision_score(y, test.rule_score):.3f}   "
          f"ROC-AUC {roc_auc_score(y, test.rule_score):.3f}")
    print(f"  challenger  PR-AUC {average_precision_score(y, prob):.3f}   ROC-AUC {roc_auc_score(y, prob):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--csv", default="data/fra_synthetic_claims_1500.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    df = pd.read_csv(args.csv)
    rule_score = contract_check(df)
    detection_metrics(df.fraud_confirmed, rule_score)
    shadow_challenger(df, rule_score)


if __name__ == "__main__":
    main()
