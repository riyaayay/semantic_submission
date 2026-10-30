"""Fast CPU tests: transliteration, cleaning, blocking, decision rules, metric."""
import numpy as np
import polars as pl

from blocking import MultiPathBlocker, extract_record_features
from decision import assign_expected_f, assign_threshold, best_expected_f_set, em_prior_shift
from metrics import compute_entity_f05
from translit import indic_to_latin


def test_transliteration_across_scripts():
    assert indic_to_latin("मॉडर्न फाइनेंस") == "modarn phaainens"
    assert indic_to_latin("হরি ফাউন্ডেশন") == "hari phaaundeshan"   # Bengali
    assert indic_to_latin("சென்னை") == "chennai"                    # Tamil
    assert indic_to_latin("Plain ASCII") == "Plain ASCII"


def test_cleaning_folds_accents_and_acronyms():
    df = pl.DataFrame({"entity_id": ["S2-1", "S2-2"], "business_name": ["Société Générale S.A.S.", "Smith & Sons"],
                       "business_address": ["12 R. de la Paix", None], "country": ["France", "US"]})
    out = extract_record_features(df)
    assert out["name_clean"].to_list() == ["societe generale sas", "smith sons"]
    assert out["words_2"].to_list() == ["societe_generale", "smith_sons"]   # no empty tokens


def _frame(ids, names, addrs, country="France"):
    return pl.DataFrame({"entity_id": ids, "business_name": names, "business_address": addrs,
                         "country": [country] * len(ids)})


def test_blocker_empty_country_and_cap_ranking():
    empty = pl.DataFrame({c: [] for c in ["entity_id", "business_name", "business_address", "country"]},
                         schema={c: pl.Utf8 for c in ["entity_id", "business_name", "business_address", "country"]})
    b = MultiPathBlocker()
    b.build_index_for_country(extract_record_features(empty))
    assert b.query_s1_for_country(extract_record_features(empty, is_s1=True)) == {}

    targets = _frame(["S2-1", "S2-2", "S2-3", "S3-9"],
                     ["Zzz Lumiere", "Yyy Lumiere", "Xxx Lumiere", "Société Lumière"],
                     ["1 A Rd", "2 B Rd", "3 C Rd", "12 Rue Caumartin 75009 Paris"])
    s1 = _frame(["S1-1"], ["Societe Lumiere"], ["12 rue Caumartin, 75009 Paris"])
    b = MultiPathBlocker(max_per_s1=1)
    b.build_index_for_country(extract_record_features(targets))
    # over the cap, the target sharing the most keys wins (not the smallest ID)
    assert b.query_s1_for_country(extract_record_features(s1, is_s1=True)) == {"S1-1": {"S3-9"}}


def test_common_name_key_is_refined_not_truncated():
    n = 120   # > max_per_key: the shared name key becomes a hub
    ids = [f"S2-{i:04d}" for i in range(n)] + ["S3-true"]
    names = ["Global Traders"] * (n + 1)
    addrs = [f"{i} Street {i} Town{i}" for i in range(n)] + ["77 Marine Drive Mumbai"]
    b = MultiPathBlocker()
    b.build_index_for_country(extract_record_features(_frame(ids, names, addrs, "India")))
    got = b.query_s1_for_country(extract_record_features(_frame(["S1-1"], ["Global Traders"], ["77 Marine Drive, Mumbai"], "India"), is_s1=True))
    assert "S3-true" in got["S1-1"]


def test_decision_rules():
    assert best_expected_f_set(np.array([0.05, 0.02])) == 0
    assert best_expected_f_set(np.array([0.95, 0.9, 0.1])) == 2
    pairs = [("A", "t1"), ("A", "t2"), ("B", "t1")]
    p = np.array([0.9, 0.4, 0.97])
    thr = assign_threshold(pairs, p, 0.5)
    assert thr["B"] == {"t1"} and not thr.get("A")          # t1 goes only to its best S1
    assert assign_expected_f(pairs, p)["B"] == {"t1"}
    adj, prior = em_prior_shift(np.array([0.9, 0.1, 0.1, 0.1]), train_prior=0.5)
    assert 0 < prior < 0.5 and adj.shape == (4,)


def test_token_signatures_identical_across_processes():
    """Bigram signatures must not depend on Python's per-process hash seed (train vs inference runs)."""
    import os
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    code = ("import sys; sys.path.insert(0, r'%s'); from gpu_features import _token_shape_features as f; "
            "r = f(['societe', 'generale', 'boulangerie', 'raaj']); print([x.tolist() for x in r])" % src)
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "12345")]
    assert outs[0] and outs[0] == outs[1]


def test_official_metric_example():
    assert abs(compute_entity_f05({"a", "b"}, {"a", "x", "b"}) - 0.7142857) < 1e-6
    assert compute_entity_f05(set(), set()) == 1.0
    assert compute_entity_f05(set(), {"a"}) == 0.0
