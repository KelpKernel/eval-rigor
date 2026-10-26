#!/usr/bin/env python3
"""eval_check.py —— 判断一个 A/B 评测结论值不值得信。

零依赖（只用标准库）。这是 eval-rigor 这个 skill 的全部计算内核：
**所有结论都必须落在这个脚本打印的数字上，不允许模型凭感觉判断。**

三种用法：

  1) 只知道题数和前后准确率
     python scripts/eval_check.py --n 20 --before 0.70 --after 0.85

  2) 有配对结果（只有 A 对 b 次、只有 B 对 c 次）
     python scripts/eval_check.py --b 3 --c 6 --n 20

  3) 想知道"我的规模能检出多大差异"
     python scripts/eval_check.py --n 60 --target-delta 0.05

  4) 同一题多次采样，想拆分方差来源
     python scripts/eval_check.py --variance "t1:110;t2:111;t3:000"

参数说明：
  --discordance  不一致率（两种方案一对一个错的比例）。实测范围 0.18–0.30，
                 不知道时按 0.25 估。样本量计算对它有依赖，所以脚本会打印出来。
  --power        统计功效，默认 0.8（业界惯例）
  --alpha        显著性水平，默认 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from statistics import NormalDist, mean

NORMAL = NormalDist()


# --------------------------------------------------------------------------
# 统计核心（与 agenteval/stats.py 同源，这里内联一份以保持零依赖）
# --------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """精确 McNemar 检验（双侧二项检验）。b/c 是两种不一致的方向。"""
    if b < 0 or c < 0:
        raise ValueError("b / c 必须非负")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    if n <= 0:
        raise ValueError("n 必须为正")
    z = NORMAL.inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def required_sample_size(
    baseline: float, target: float, discordance: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """配对设计所需题目数（Connor 1987）。

        n = [z_{α/2}·√π + z_β·√(π − δ²)]² / δ²
    """
    delta = abs(target - baseline)
    if delta <= 0:
        raise ValueError("target 与 baseline 相同，没有差异需要检出")
    if not 0 < discordance <= 1:
        raise ValueError("discordance 必须在 (0, 1]")
    if delta**2 >= discordance:
        raise ValueError(
            f"δ²={delta**2:.4f} 不小于不一致率 {discordance}：假设不自洽"
            "（差异比不一致对还多），请调高 --discordance"
        )
    z_alpha = NORMAL.inv_cdf(1 - alpha / 2)
    z_beta = NORMAL.inv_cdf(power)
    term = z_alpha * math.sqrt(discordance) + z_beta * math.sqrt(discordance - delta**2)
    return math.ceil(term**2 / delta**2)


def detectable_delta(n: int, discordance: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """反过来问：给定题目数，最小能检出多大的准确率差异（二分求解）。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    if not 0 < discordance <= 1:
        raise ValueError("discordance 必须在 (0, 1]")
    z_alpha = NORMAL.inv_cdf(1 - alpha / 2)
    z_beta = NORMAL.inv_cdf(power)
    low, high = 1e-6, math.sqrt(discordance) - 1e-9
    for _ in range(200):
        mid = (low + high) / 2
        term = z_alpha * math.sqrt(discordance) + z_beta * math.sqrt(discordance - mid**2)
        if term**2 / mid**2 > n:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def variance_components(repeated: dict[str, list[int]]) -> dict:
    """把多次采样的方差拆成"任务难度"与"采样噪声"。"""
    lengths = {len(v) for v in repeated.values()}
    if len(lengths) != 1:
        raise ValueError(f"每题的采样次数必须相同，收到 {sorted(lengths)}")
    k = lengths.pop()
    per_task = [mean(v) for v in repeated.values()]
    observed = float(__import__("statistics").variance(per_task)) if len(per_task) > 1 else 0.0
    sampling = mean(p * (1 - p) / k for p in per_task)
    task = max(observed - sampling, 0.0)
    return {
        "tasks": len(per_task),
        "samples_per_task": k,
        "mean_accuracy": mean(per_task),
        "observed_variance": observed,
        "task_variance": task,
        "sampling_variance": sampling,
        "task_share": (task / observed) if observed > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# 输入解析与输出
# --------------------------------------------------------------------------
def parse_rate(value: str) -> float:
    """接受 0.85 / '85%' / 85 三种写法。"""
    text = str(value).strip().rstrip("%")
    number = float(text)
    if number > 1.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"准确率必须在 0–1（或 0–100%）之间，收到 {value}")
    return number


def parse_variance(spec: str) -> dict[str, list[int]]:
    """格式：t1:110;t2:111;t3:000 —— 冒号后是每题每次采样的对错。"""
    result: dict[str, list[int]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"方差输入格式错误：{chunk}（应为 任务:101）")
        name, flags = chunk.split(":", 1)
        cleaned = [ch for ch in flags.strip() if ch in "01"]
        if not cleaned:
            raise ValueError(f"{name} 没有有效的对错序列")
        result[name] = [int(ch) for ch in cleaned]
    if not result:
        raise ValueError("方差输入为空")
    return result


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def report_planned(n: int, discordance: float, power: float, alpha: float) -> str:
    floor = detectable_delta(n, discordance, alpha=alpha, power=power)
    lines = [
        "== 当前规模的分辨率 ==",
        f"题目数            {n}",
        f"不一致率假设      {pct(discordance)}",
        f"可检出下限        {pct(floor)}（功效 {power:.0%}，α={alpha}）",
        "",
        "结论：",
        f"  {n} 道题只能稳定检出 {pct(floor)} 以上的差异。",
        f"  小于这个数的改进，无论看起来多漂亮，都无法与采样噪声区分。",
        "",
        "参考：",
        "  题目数 → 可检出下限",
    ]
    for candidate in (60, 100, 200, 400, 800):
        lines.append(f"    {candidate:<6}{pct(detectable_delta(candidate, discordance, alpha=alpha, power=power))}")
    return "\n".join(lines)


def report_compare(
    n: int, baseline: float, target: float, discordance: float, power: float, alpha: float
) -> str:
    delta = target - baseline
    floor = detectable_delta(n, discordance, alpha=alpha, power=power)
    observed_abs = abs(delta)
    lines = [
        "== 输入 ==",
        f"题目数            {n}",
        f"准确率            {pct(baseline)} → {pct(target)}",
        f"观察差异          {delta:+.1%}",
        f"不一致率假设      {pct(discordance)}",
        "",
        "== 计算 ==",
        f"可检出下限        {pct(floor)}（功效 {power:.0%}，α={alpha}）",
    ]
    if observed_abs >= floor:
        lines += ["", "== 结论 ==", f"  观察差异 {pct(observed_abs)} ≥ 可检出下限 {pct(floor)}：有足够分辨率。"]
    else:
        try:
            needed = required_sample_size(baseline, target, discordance, alpha=alpha, power=power)
            need_text = f"要稳定检出这个差异，需要约 {needed} 题（当前 {n} 题）。"
        except ValueError as exc:
            need_text = f"无法给出所需题数：{exc}"
        lines += [
            "",
            "== 结论 ==",
            f"  观察差异 {pct(observed_abs)} < 可检出下限 {pct(floor)}：",
            "  这个规模下无法与噪声区分。",
            f"  {need_text}",
            "",
            "  只能这样表述：",
            f"  「观察到 {delta:+.1%} 的差异，但当前样本量不足以支持该差异为真。」",
            "  不要写成「提升了 X 个点」。",
        ]
    return "\n".join(lines)


def report_paired(n: int, b: int, c: int, baseline: float | None, target: float | None) -> str:
    p_value = mcnemar_exact(b, c)
    lines = ["== 配对检验（精确 McNemar）=="]
    if n is not None:
        lines.append(f"题目数            {n}")
    lines += [
        f"只有 A 对          {b}",
        f"只有 B 对          {c}",
        f"不一致对          {b + c}",
    ]
    if baseline is not None and target is not None:
        lines.append(f"准确率变化        {target - baseline:+.1%}")
    lines += [
        "",
        f"McNemar p = {p_value:.4f}",
        "",
        "== 结论 ==",
    ]
    if b + c == 0:
        lines.append("  两种方案在每一题上表现相同，没有证据表明有差异。")
    elif p_value < 0.05:
        better = "B" if c > b else "A"
        lines.append(f"  在 α=0.05 水平上显著，{better} 更好。")
        if n is not None:
            floor = detectable_delta(n, 0.25)
            delta = abs((c - b)) / n if n else 0
            if delta < floor:
                lines.append(
                    f"  但注意：观察差异 {pct(delta)} 低于该规模的可检出下限 {pct(floor)}，"
                    "这次显著有运气成分，换一批采样可能翻转。"
                )
    else:
        lines.append("  差异不显著：当前样本量不支持「变了」这个判断。")
        if n:
            lines.append(f"  这个规模只能检出 {pct(detectable_delta(n, 0.25))} 以上的差异。")
    return "\n".join(lines)


def report_variance(repeated: dict[str, list[int]]) -> str:
    parts = variance_components(repeated)
    lines = [
        "== 方差分解 ==",
        f"题目数 × 采样次数  {parts['tasks']} × {parts['samples_per_task']}",
        f"平均准确率        {pct(parts['mean_accuracy'])}",
        "",
        f"总方差 {parts['observed_variance']:.4f} = "
        f"任务难度 {parts['task_variance']:.4f} + 采样噪声 {parts['sampling_variance']:.4f}",
        f"任务难度占比      {parts['task_share']:.0%}",
        "",
        "== 判读 ==",
    ]
    if parts["task_share"] >= 0.7:
        lines += [
            "  任务难度占大头 → 题目之间差别大，**加题**才是提高分辨率的手段。",
            "  加采样次数只会让同一题的数字更稳，不会让你检出更小的差异。",
        ]
    else:
        lines += [
            "  采样噪声占大头 → 这个配置本身的指标会漂。",
            "  先降低随机性（温度、多次采样取多数），再加题才有效。",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="判断一个 A/B 评测结论值不值得信（零依赖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n", type=int, help="题目数")
    parser.add_argument("--before", help="基线准确率（0.70 / 70 / '70%%' 均可）")
    parser.add_argument("--after", help="改动后准确率")
    parser.add_argument("--b", type=int, help="只有 A 对的题数")
    parser.add_argument("--c", type=int, help="只有 B 对的题数")
    parser.add_argument("--target-delta", help="想检出的差异大小（如 0.05）")
    parser.add_argument("--variance", help="方差分解输入，格式 t1:110;t2:111")
    parser.add_argument("--discordance", default="0.25", help="不一致率，默认 0.25")
    parser.add_argument("--power", type=float, default=0.8, help="统计功效，默认 0.8")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平，默认 0.05")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    try:
        discordance = parse_rate(args.discordance)

        if args.variance:
            repeated = parse_variance(args.variance)
            text = report_variance(repeated)
            payload = variance_components(repeated)
        elif args.b is not None and args.c is not None:
            baseline = parse_rate(args.before) if args.before else None
            target = parse_rate(args.after) if args.after else None
            text = report_paired(args.n, args.b, args.c, baseline, target)
            payload = {"mcnemar_p": mcnemar_exact(args.b, args.c), "b": args.b, "c": args.c}
        elif args.n and args.before and args.after:
            baseline = parse_rate(args.before)
            target = parse_rate(args.after)
            text = report_compare(args.n, baseline, target, discordance, args.power, args.alpha)
            payload = {
                "n": args.n,
                "observed_delta": target - baseline,
                "detectable_delta": detectable_delta(args.n, discordance, alpha=args.alpha, power=args.power),
            }
            try:
                payload["required_n"] = required_sample_size(
                    baseline, target, discordance, alpha=args.alpha, power=args.power
                )
            except ValueError:
                payload["required_n"] = None
        elif args.n and args.target_delta:
            delta = parse_rate(args.target_delta)
            payload = {
                "n": args.n,
                "target_delta": delta,
                "required_n": required_sample_size(1 - delta, 1.0, discordance, alpha=args.alpha, power=args.power),
                "detectable_delta": detectable_delta(args.n, discordance, alpha=args.alpha, power=args.power),
            }
            text = report_planned(args.n, discordance, args.power, args.alpha)
            text += (
                f"\n\n要检出 {pct(delta)} 的差异，需要约 {payload['required_n']} 题。"
            )
        else:
            parser.print_help()
            return 2
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
