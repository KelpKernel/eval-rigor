"""eval_check.py 的测试。

这个脚本是整个 skill 的计算内核——它算错，skill 就会给出错误的判断，
所以每个公式都要对住手算值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_check as ec  # noqa: E402


# ---------------- 输入解析 ----------------
@pytest.mark.parametrize(
    "raw,expected",
    [("0.85", 0.85), ("85%", 0.85), ("85", 0.85), ("0.7", 0.7), ("100", 1.0), ("0", 0.0)],
)
def test_parse_rate_accepts_three_notations(raw, expected):
    assert ec.parse_rate(raw) == pytest.approx(expected)


def test_parse_rate_rejects_out_of_range():
    with pytest.raises(ValueError):
        ec.parse_rate("-5")


def test_parse_variance():
    parsed = ec.parse_variance("t1:110; t2:0 0 0 ;t3:1,1,1")
    assert parsed == {"t1": [1, 1, 0], "t2": [0, 0, 0], "t3": [1, 1, 1]}


def test_parse_variance_rejects_garbage():
    with pytest.raises(ValueError):
        ec.parse_variance("t1-abc")


# ---------------- McNemar ----------------
def test_mcnemar_balanced_is_one():
    assert ec.mcnemar_exact(9, 9) == pytest.approx(1.0, abs=1e-9)


def test_mcnemar_known_value():
    assert ec.mcnemar_exact(0, 10) == pytest.approx(2 / 2**10, rel=1e-9)


def test_mcnemar_datagent_case():
    """真实数据：flash 独对 7、deepseek 独对 11 → p≈0.48。"""
    assert ec.mcnemar_exact(7, 11) == pytest.approx(0.4807, abs=5e-4)


def test_mcnemar_no_discordance():
    assert ec.mcnemar_exact(0, 0) == 1.0


# ---------------- 样本量 ----------------
def test_required_sample_size_matches_manual_value():
    """δ=0.10, π=0.25 → n≈194（Connor 公式手算）。"""
    assert ec.required_sample_size(0.48, 0.58, 0.25) == pytest.approx(194, abs=3)


def test_required_sample_size_grows_with_smaller_delta():
    small = ec.required_sample_size(0.4, 0.45, 0.25)
    big = ec.required_sample_size(0.4, 0.60, 0.25)
    assert small > big


def test_required_sample_size_rejects_inconsistent_assumption():
    with pytest.raises(ValueError):
        ec.required_sample_size(0.0, 0.90, 0.20)


def test_big_delta_needs_fewer_questions_than_small():
    """20 题的规模检出不了 15 个点，但 60 题能检出 30 个点。"""
    assert ec.detectable_delta(20, 0.25) > 0.15
    assert ec.detectable_delta(60, 0.25) < 0.30


def test_detectable_delta_and_required_sample_size_are_inverse():
    n = ec.required_sample_size(0.48, 0.58, 0.25)
    assert ec.detectable_delta(n, 0.25) == pytest.approx(0.10, abs=5e-3)


# ---------------- 方差分解 ----------------
def test_variance_all_stable_tasks_have_zero_sampling_noise():
    parts = ec.variance_components({"a": [1, 1, 1], "b": [0, 0, 0], "c": [1, 1, 1]})
    assert parts["sampling_variance"] == pytest.approx(0.0)
    assert parts["task_share"] == pytest.approx(1.0)


def test_variance_noise_dominated():
    parts = ec.variance_components(
        {"a": [1, 0, 1, 0, 1, 0], "b": [0, 1, 0, 1, 0, 1], "c": [1, 0, 0, 1, 0, 1]}
    )
    assert parts["sampling_variance"] > parts["task_variance"]


def test_variance_requires_equal_sample_counts():
    with pytest.raises(ValueError):
        ec.variance_components({"a": [1], "b": [1, 0]})


# ---------------- 报告文案 ----------------
def test_report_compare_flags_noise_and_gives_needed_n():
    text = ec.report_compare(20, 0.70, 0.85, 0.25, 0.8, 0.05)
    assert "无法与噪声区分" in text
    assert "需要约" in text and "题" in text
    assert "不要写成" in text  # 明确禁止的表述


def test_report_compare_passes_when_delta_exceeds_floor():
    text = ec.report_compare(400, 0.60, 0.75, 0.25, 0.8, 0.05)
    assert "有足够分辨率" in text


def test_report_paired_insignificant():
    text = ec.report_paired(20, 3, 6, 0.70, 0.85)
    assert "不显著" in text


def test_report_paired_significant():
    text = ec.report_paired(60, 2, 15, 0.60, 0.82)
    assert "显著" in text and "更好" in text


def test_report_paired_identical_has_no_evidence():
    text = ec.report_paired(20, 0, 0, 0.7, 0.7)
    assert "没有证据" in text


def test_report_variance_tells_you_to_add_tasks_or_sampling():
    stable = ec.report_variance({"a": [1, 1, 1], "b": [0, 0, 0], "c": [1, 1, 1]})
    assert "加题" in stable

    noisy = ec.report_variance(
        {"a": [1, 0, 1, 0], "b": [0, 1, 0, 1], "c": [1, 0, 0, 1], "d": [0, 1, 1, 0]}
    )
    assert "随机性" in noisy


def test_report_planned_lists_reference_table():
    text = ec.report_planned(60, 0.25, 0.8, 0.05)
    assert "可检出下限" in text
    for n in ("60", "100", "200", "400"):
        assert n in text


# ---------------- CLI 端到端 ----------------
def test_cli_compare_mode(capsys):
    code = ec.main(["--n", "20", "--before", "0.70", "--after", "0.85"])
    assert code == 0
    assert "可检出下限" in capsys.readouterr().out


def test_cli_paired_mode(capsys):
    code = ec.main(["--b", "3", "--c", "6", "--n", "20"])
    assert code == 0
    out = capsys.readouterr().out
    assert "McNemar" in out and "0.5078" in out


def test_cli_json_mode(capsys):
    code = ec.main(["--n", "60", "--before", "0.48", "--after", "0.55", "--json"])
    assert code == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 60
    assert payload["required_n"] > 60


def test_cli_target_delta_mode(capsys):
    code = ec.main(["--n", "60", "--target-delta", "0.05"])
    assert code == 0
    assert "需要约" in capsys.readouterr().out


def test_cli_variance_mode(capsys):
    code = ec.main(["--variance", "t1:111;t2:000;t3:110"])
    assert code == 0
    assert "方差" in capsys.readouterr().out


def test_cli_rejects_bad_arguments(capsys):
    assert ec.main(["--n", "20", "--before", "0.7", "--after", "0.85", "--discordance", "0"]) != 0


def test_cli_without_arguments_prints_help(capsys):
    assert ec.main([]) == 2
