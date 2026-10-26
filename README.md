# eval-rigor

[![CI](https://github.com/KelpKernel/eval-rigor/actions/workflows/ci.yml/badge.svg)](https://github.com/KelpKernel/eval-rigor/actions/workflows/ci.yml)

**当有人说「我改了点东西，准确率从 70% 涨到 85%」时，先算一下这个结论值不值得信。**

一个 Agent Skill：让 AI 助手在你报出评测数字的那一刻先做统计判断，而不是替你叫好。

## 它拦下的是什么

最常见的场景：

> 「我把 temperature 从 0.7 降到 0.3，跑了 20 道题，准确率从 70% 涨到 85%，是不是有效？」

装上这个 skill 之后，回答会变成：

```
可检出下限   29.5%（80% 功效，α=0.05）
结论：观察差异 15.0% < 可检出下限 29.5% —— 这个规模下无法与噪声区分。
      要稳定检出这个差异，需要约 85 题（当前 20 题）。
```

## 装上之后

| 你说 | 它会做的 |
|---|---|
| 「改了 prompt，准确率提升了」 | 问清题数和采样次数，算可检出下限，判断是不是噪声 |
| 「A 对 B 错的 3 题，A 错 B 对的 6 题」 | 直接跑精确 McNemar，给 p 值和区间 |
| 「我需要多少道题才能测出来」 | 用 Connor 公式给出具体题数 |
| 「同一题跑了几次，结果不稳」 | 把方差拆成任务难度 vs 采样噪声，告诉你该加题还是加采样 |

## 安装

```bash
npx skills add https://github.com/KelpKernel/eval-rigor --skill eval-rigor
```

或者手动：把 `SKILL.md` 和 `scripts/` 复制到你的 skills 目录
（Reasonix / Claude Code 一般是 `~/.agents/skills/eval-rigor/`）。

## 依赖

**脚本零第三方依赖**（只用 Python 标准库）。需要 Python 3.10+。

```bash
python scripts/eval_check.py --help
```

## 为什么数字必须来自脚本

一个判断评测可信度的工具，如果自己的关键数字靠模型心算，它就成了它要批判的那种东西。
所以 `SKILL.md` 里明确写着：**不要跳过脚本去自己估算**。

脚本里的公式：

- **配对检验**：精确 McNemar（二项检验，不做卡方近似——评测集通常只有几十到几百题）
- **样本量**：Connor (1987) 配对比例检验公式，`n = [z_α√π + z_β√(π−δ²)]² / δ²`
- **区间**：Wilson 得分区间（k=0 或 k=n 时不会退化成零宽）
- **方差分解**：任务间方差 − 二项采样噪声 = 真实任务方差

## 相关的另一个项目

计算内核来自 [agenteval](https://github.com/KelpKernel/agenteval) —— 那个库是
「跑评测 + 出报告 + CI 门禁」，这个 skill 是「在任何对话里就地判断结论」。

## 测试

35 个测试，`python -m pytest tests -q`。
