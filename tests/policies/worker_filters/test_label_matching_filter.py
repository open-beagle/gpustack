from gpustack.policies.worker_filters.label_matching_filter import label_matching


def test_label_matching_returns_false_when_current_labels_are_missing():
    assert label_matching({"os": "linux"}, None) is False
