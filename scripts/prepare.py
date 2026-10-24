"""Step 1: build a cohort's analysis table and lock away the test participants.

    python prepare.py --cohort framingham|nhanes

About 20% of PARTICIPANTS (not rows) go to the test set, stratified on whether
the person has any event. All later tuning uses the training participants only.
"""
import argparse
import json

from sklearn.model_selection import StratifiedGroupKFold

import cohorts
from config import GROUP, RANDOM_STATE, TARGET, TEST_FRACTION, dirs


def main(name):
    c, out = cohorts.get(name), dirs(name)
    df, log = c.build()
    person_event = df.groupby(GROUP)[TARGET].transform("max")
    sgkf = StratifiedGroupKFold(n_splits=round(1 / TEST_FRACTION), shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(sgkf.split(df, person_event, groups=df[GROUP]))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    assert not set(train[GROUP]) & set(test[GROUP]), "participant leaked across split"
    train.to_csv(out["data"] / "train.csv", index=False)
    test.to_csv(out["data"] / "test.csv", index=False)
    for part, d in (("all", df), ("train", train), ("test", test)):
        log[part] = {"rows": int(len(d)), "participants": int(d[GROUP].nunique()), "events": int(d[TARGET].sum()),
                     "event_rate": round(float(d[TARGET].mean()), 4)}
    log["missing_share"] = {f: round(float(df[f].isna().mean()), 4) for f in c.FEATURES if df[f].isna().any()}
    (out["rep"] / "data_summary.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    main(ap.parse_args().cohort)
