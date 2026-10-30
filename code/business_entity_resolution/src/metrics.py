"""
Evaluation metric for Amazon ML Challenge 2026: Macro F_0.5 per S1 entity.

Formula:
  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singleton Rule:
  A Source 1 entity with no true matches (singleton):
    - Score = 1.0 if predicted match list is empty
    - Score = 0.0 if any match is predicted
"""

from typing import Dict, Set, List, Union


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Computes F_0.5 score for a single Source 1 entity."""
    if not true_matches:
        # Singleton entity
        return 1.0 if not pred_matches else 0.0

    if not pred_matches:
        # True matches exist, but nothing predicted
        return 0.0

    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Computes Macro F_0.5 across all S1 entities in ground_truth.
    
    Args:
        ground_truth: mapping of {s1_id: set_of_matched_ids}
        predictions: mapping of {s1_id: set_of_predicted_matched_ids}
        
    Returns:
        Dictionary with overall 'macro_f05', 'singleton_f05', 'non_singleton_f05',
        'singleton_count', and 'non_singleton_count'.
    """
    total_score = 0.0
    singleton_score = 0.0
    non_singleton_score = 0.0
    singleton_count = 0
    non_singleton_count = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        total_score += score

        if not true_set:
            singleton_count += 1
            singleton_score += score
        else:
            non_singleton_count += 1
            non_singleton_score += score

    n = len(ground_truth)
    return {
        "macro_f05": total_score / n if n > 0 else 0.0,
        "singleton_f05": singleton_score / singleton_count if singleton_count > 0 else 0.0,
        "non_singleton_f05": non_singleton_score / non_singleton_count if non_singleton_count > 0 else 0.0,
        "singleton_count": singleton_count,
        "non_singleton_count": non_singleton_count,
        "total_entities": n
    }


if __name__ == "__main__":
    # Test with official example from problem statement:
    # S1-00001 predicted: [S2-00047, S2-00193, S3-00812]
    # Ground truth: [S2-00047, S3-00812]
    # Precision = 2/3 = 0.667, Recall = 2/2 = 1.0 -> F0.5 = 0.714
    example_gt = {"S1-00001": {"S2-00047", "S3-00812"}, "S1-00002": set()}
    example_pred = {
        "S1-00001": {"S2-00047", "S2-00193", "S3-00812"},
        "S1-00002": set()  # Correct singleton -> 1.0
    }
    s1_score = compute_entity_f05(example_gt["S1-00001"], example_pred["S1-00001"])
    print(f"Example S1-00001 F_0.5: {s1_score:.3f} (Expected: 0.714)")
    assert abs(s1_score - 0.7142857) < 1e-4

    results = compute_macro_f05(example_gt, example_pred)
    print(f"Overall Macro F_0.5: {results['macro_f05']:.4f}")
    assert abs(results["macro_f05"] - (0.7142857 + 1.0) / 2) < 1e-4
    print("All metric tests passed successfully!")
