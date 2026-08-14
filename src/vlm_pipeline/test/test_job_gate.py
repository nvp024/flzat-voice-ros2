from vlm_pipeline.job_gate import VlmJobGate


def test_gate_allows_only_one_reserved_request() -> None:
    gate = VlmJobGate()
    assert gate.try_reserve()
    assert gate.busy
    assert not gate.try_reserve()
    gate.release()
    assert not gate.busy
    assert gate.try_reserve()
