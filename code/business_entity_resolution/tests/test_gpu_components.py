"""GPU tests: bit-parallel string kernels match rapidfuzz; block-wise v8 features equal one-pass."""
import random

import numpy as np
import polars as pl
import pytest
import torch
from rapidfuzz.distance import LCSseq, Levenshtein

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def _random_pairs(n, max_len, seed=0):
    rng = random.Random(seed)
    alpha = "abcdefghij 0123"
    a = ["".join(rng.choice(alpha) for _ in range(rng.randint(0, max_len))) for _ in range(n)]
    b = [s if rng.random() < 0.3 else "".join(rng.choice(alpha) for _ in range(rng.randint(0, max_len))) for s in a]
    return a, b


@pytest.mark.parametrize("max_len,words", [(63, 1), (189, 3)])
def test_lcs_matches_rapidfuzz(max_len, words):
    from gpu_strings import build_peq, encode_strings, lcs_length
    a, b = _random_pairs(3000, max_len)
    sa, la = encode_strings(a, max_len)
    sb, _ = encode_strings(b, max_len)
    peq = build_peq(torch.as_tensor(sa, device="cuda"), words)
    got = lcs_length(peq, torch.as_tensor(la, device="cuda"), torch.as_tensor(sb, device="cuda")).cpu().numpy()
    assert (got == np.array([LCSseq.similarity(x, y) for x, y in zip(a, b)])).all()


def test_levenshtein_matches_rapidfuzz():
    from gpu_strings import build_peq, encode_strings, levenshtein
    a, b = _random_pairs(3000, 63, seed=1)
    sa, la = encode_strings(a, 63)
    sb, lb = encode_strings(b, 63)
    peq = build_peq(torch.as_tensor(sa, device="cuda"), 1)
    got = levenshtein(peq, torch.as_tensor(la, device="cuda"), torch.as_tensor(sb, device="cuda"),
                      torch.as_tensor(lb, device="cuda")).cpu().numpy()
    assert (got == np.array([Levenshtein.distance(x, y) for x, y in zip(a, b)])).all()


def test_v9_semantic_layer_from_shipped_model():
    import v8_pipeline as vp
    from blocking import extract_record_features

    tg = pl.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3"],
                       "business_name": ["Central Pharmacy", "Garage Martin", "Boulangerie Dupont"],
                       "business_address": ["4 Place Bellecour Lyon", "9 Rue Lafayette Paris", "12 Rue de la Paix Paris"],
                       "country": ["France"] * 3})
    s1 = pl.DataFrame({"entity_id": ["S1-1"], "business_name": ["Pharmacie Centrale"],
                       "business_address": ["4 place Bellecour, Lyon"], "country": ["France"]})
    pt, ps = extract_record_features(tg), extract_record_features(s1, is_s1=True)
    cands, Es, Et = vp.semantic_candidates(ps, pt, k=1)
    assert cands == {"S1-1": {"S2-1"}}          # meaning match across languages
    pairs = [("S1-1", t) for t in ["S2-1", "S2-2", "S2-3"]]
    X = vp.country_features(ps, pt, pairs, version="v9", sem=(Es, Et))
    names = vp.FEATURE_NAMES_V9
    assert X.shape == (3, len(names)) and np.isfinite(X).all()
    cos = X[:, names.index("name_sem_cos")]
    assert cos[0] > 0.9 and cos[0] > cos[1] and cos[0] > cos[2]
    assert X[0, names.index("grp_name_sem_gap")] == 0.0


def test_topm_groups_label_free_ranking():
    from judge import RANK_FEATURES, topm_groups
    names = list(RANK_FEATURES)
    ia = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    X = np.zeros((5, len(names)), dtype=np.float32)
    X[:, 0] = [0.1, 0.9, 0.5, 0.2, 0.8]
    g = topm_groups(ia, X, names, m=2)
    assert g.tolist() == [[1, 2], [4, 3]]


def test_judge_learns_set_decisions():
    from judge import JudgeConfig, JudgeData, predict_judge, train_judge
    rng = np.random.default_rng(0)
    N, m, Fn, Fe = 600, 6, 4, 13
    nodes = rng.normal(size=(N, m, Fn)).astype(np.float32)
    y = (nodes[..., 0] > 0.8).astype(np.float32)          # learnable rule
    edges = np.zeros((N, m, m, Fe), dtype=np.float32)
    mask = np.ones((N, m), dtype=bool)
    mask[:, -1] = False
    y[:, -1] = 0
    d = JudgeData(torch.as_tensor(nodes, dtype=torch.float16), torch.as_tensor(edges, dtype=torch.float16),
                  torch.as_tensor(mask), torch.as_tensor(y), torch.as_tensor((y.sum(1) > 0).astype(np.float32)))
    model = train_judge(d.subset(np.arange(500)), d.subset(np.arange(500, 600)), JudgeConfig(n_node_features=Fn),
                        epochs=15, batch=100, log=lambda *_: None)
    out = predict_judge(model, d.subset(np.arange(500, 600)))
    acc = ((out["p"] > 0.5) == (y[500:] > 0.5))[mask[500:]].mean()
    assert acc > 0.9 and np.isfinite(out["p"]).all() and ((out["p"] >= 0) & (out["p"] <= 1)).all()


def test_blockwise_features_equal_one_pass():
    import v8_pipeline as vp
    from blocking import MultiPathBlocker, extract_record_features
    from gpu_features import build_country_vocab, encode_bank, features_v8

    rng = random.Random(3)
    words = ["boulangerie", "dupont", "pharmacie", "centrale", "garage", "martin", "societe", "lumiere", "cafe", "paris"]
    def rec(i, p):
        return (f"{p}-{i}", " ".join(rng.sample(words, 2)), f"{rng.randint(1, 30)} rue {rng.choice(words)} paris")
    tg = pl.DataFrame([rec(i, "S2") for i in range(400)], schema=["entity_id", "business_name", "business_address"], orient="row")
    s1 = pl.DataFrame([rec(i, "S1") for i in range(60)], schema=["entity_id", "business_name", "business_address"], orient="row")
    tg, s1 = tg.with_columns(pl.lit("France").alias("country")), s1.with_columns(pl.lit("France").alias("country"))
    pt, ps = extract_record_features(tg), extract_record_features(s1, is_s1=True)
    b = MultiPathBlocker()
    b.build_index_for_country(pt)
    s_ids, t_ids, ia, ib = vp.pair_arrays(b.query_s1_for_country(ps))
    assert len(ia) > 0

    voc = build_country_vocab(pt)
    bs = encode_bank(vp._rows_in_order(ps, "s1_id", s_ids), voc, "s1_id", pattern_side=True)
    bt = encode_bank(vp._rows_in_order(pt, "target_id", t_ids), voc, "target_id", pattern_side=False)
    ref = features_v8(bs, bt, torch.as_tensor(ia.astype(np.int64), device="cuda"),
                      torch.as_tensor(ib.astype(np.int64), device="cuda")).cpu().numpy()
    old = vp.FEATURE_CHUNK
    vp.FEATURE_CHUNK = 200
    try:
        out = np.empty_like(ref)
        vp.features_into(ps, pt, s_ids, t_ids, ia, ib, out)
    finally:
        vp.FEATURE_CHUNK = old
    assert np.isfinite(ref).all()
    assert np.array_equal(out, ref)
