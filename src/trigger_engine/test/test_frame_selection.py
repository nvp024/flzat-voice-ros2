from trigger_engine.frame_selection import (
    FrameCandidate,
    relevance_window_end_ns,
    select_relevant_frame,
)


def _candidate(source: str, stamp_ns: int) -> FrameCandidate[str]:
    return FrameCandidate(source, stamp_ns, source)


def test_relevance_end_never_extends_beyond_stt_or_visual_limit() -> None:
    assert relevance_window_end_ns(1_000, 1_500, 750) == 1_500
    assert relevance_window_end_ns(1_000, 2_000, 750) == 1_750


def test_newest_valid_candidate_is_selected() -> None:
    baseline = _candidate("baseline", 1_100)
    refreshed = _candidate("refreshed", 1_650)
    motion = _candidate("motion", 1_500)

    result = select_relevant_frame(
        [baseline, refreshed, motion],
        baseline,
        window_start_ns=1_000,
        window_end_ns=1_700,
    )

    assert result.selected is refreshed
    assert result.used_baseline_fallback is False


def test_out_of_window_refresh_is_rejected() -> None:
    baseline = _candidate("baseline", 1_100)
    refreshed = _candidate("refreshed", 1_800)

    result = select_relevant_frame(
        [baseline, refreshed],
        baseline,
        window_start_ns=1_000,
        window_end_ns=1_700,
    )

    assert result.selected is baseline
    assert result.rejected_sources == ("refreshed",)


def test_baseline_is_fallback_when_all_candidates_are_outside() -> None:
    baseline = _candidate("baseline", 900)
    refreshed = _candidate("refreshed", 1_800)

    result = select_relevant_frame(
        [baseline, refreshed],
        baseline,
        window_start_ns=1_000,
        window_end_ns=1_700,
    )

    assert result.selected is baseline
    assert result.used_baseline_fallback is True
    assert result.rejected_sources == ("refreshed",)


def test_no_baseline_means_out_of_window_frames_are_not_used() -> None:
    motion = _candidate("motion", 800)

    result = select_relevant_frame(
        [motion],
        baseline=None,
        window_start_ns=1_000,
        window_end_ns=1_700,
    )

    assert result.selected is None
    assert result.rejected_sources == ("motion",)


def test_equal_timestamp_prefers_refreshed_candidate() -> None:
    baseline = _candidate("baseline", 1_500)
    motion = _candidate("motion", 1_500)
    refreshed = _candidate("refreshed", 1_500)

    result = select_relevant_frame(
        [baseline, motion, refreshed],
        baseline,
        window_start_ns=1_000,
        window_end_ns=1_700,
    )

    assert result.selected is refreshed
