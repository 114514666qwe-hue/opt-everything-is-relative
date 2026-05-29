# Section 4.2 COMPAS 复现实验说明

## 任务范围

本文件夹复现并审计论文 `Everything is Relative: Understanding Fairness with
Optimal Transport` 的 Section 4.2。原论文没有公开实验代码，因此这里采用
“公开数据 + 论文方法描述 + 多个合理实现分支”的方式重建实验。

## 数据集与过滤

- 数据来源：`https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv`
- 原始行数：7214
- 使用 ProPublica 标准过滤后的行数：6172
- 用于本报告矩阵的 Black/White 子集行数：5278
- 过滤后 Black defendants：3175
- 过滤后 White defendants：2103
- 过滤后 two-year recidivists：2809

过滤条件为：

1. `-30 <= days_b_screening_arrest <= 30`
2. `is_recid != -1`
3. `c_charge_degree != "O"`
4. `score_text != "N/A"`

这些计数和论文附录一致：6172 总样本、3175 Black、2103 White、2809 个
two-year recidivists。因此数据源和过滤方式基本可以确认。

## 实验思路

论文比较同一批个体上的两个 policy：

1. `F_true`：用 logistic regression 拟合真实 two-year recidivism。
2. `F_compas`：用 logistic regression 拟合 COMPAS 的二元风险标签。

这里把 COMPAS 的 `Low` 作为低风险，把 `Medium/High` 合并为高风险。这是
ProPublica COMPAS 分析里最常见的二分方式。

每个 policy 对个体 `x_i` 输出一个二分类概率向量：

```text
F(x_i) = [1 - p_i, p_i]
```

于是两个 policy outcome 之间的代价为：

```text
C_ij = ||F_true(x_i) - F_compas(x_j)||_2
```

在二分类情形下，这个代价和 `|p_i - p_j|` 只差一个常数因子。因此最优传输
可以用一维分位数匹配精确求解：把 `F_true` 的概率排序，把 `F_compas` 的概率
排序，然后按经验分布质量逐段匹配。这个结果与离散 OT 线性规划的最优代价一致，
但避免构造巨大的 dense coupling matrix。

得到 coupling `pi` 后，将每条传输边按 race 和 COMPAS 二元风险组聚合：

```text
WhiteLow, WhiteHigh, BlackLow, BlackHigh
```

脚本同时输出两类矩阵：

- `mass_row_pct`：只看 transport mass 的行归一化占比。
- `bias_row_pct`：看 `transport_mass * feature_distance` 的行归一化占比。

第二种更接近论文的 individual/group bias 定义。离散形式下，对 source 个体
`a_i` 的 individual bias 可写作：

```text
u(a_i) = n * sum_j pi_ij d(x_i, x_j)
```

其中 `n * pi_ij` 是第 `i` 行 coupling 的条件分布质量，`d` 是特征空间距离。
group-wise bias decomposition 则把 `u(a_i)` 按 target subgroup 分解。

## 主要结果

最接近论文 Figure 3A 第一处高亮数字的是
`criminal_only_compas_groups`。该分支用刑事历史相关变量训练 logistic model，
并用同一类变量计算 feature distance：

| 指标 | 原文 | 本复现 |
| --- | ---: | ---: |
| `WhiteLow -> BlackHigh` bias share | 43.8% | 46.94% |
| `BlackHigh -> WhiteLow` bias share | 48.3% | 8.02% |

这个分支能较好恢复“WhiteLow 与 BlackHigh 存在显著交叉映射/偏差贡献”的结构，
但不能恢复原文第二个 `BlackHigh -> WhiteLow = 48.3%` 数字。

较常规的 expanded feature 分支结果为：

| 指标 | 原文 | expanded 复现 |
| --- | ---: | ---: |
| `WhiteLow -> BlackHigh` bias share | 43.8% | 23.53% |
| `BlackHigh -> WhiteLow` bias share | 48.3% | 9.81% |

因此，Section 4.2 的定性结构可以复现，但 Figure 3A 的两个具体百分比不能在
论文给出的信息下唯一复现。

## Equal Opportunity 代理实验

论文还展示了 Figure 3B：在 COMPAS 标签上训练第三个 logistic regression，
并加入 Zafar et al. 的 equal opportunity 约束。论文没有给出约束参数、优化器、
特征矩阵和 train/test split，因此这里没有声称精确复现 Figure 3B。

为了观察趋势，脚本实现了一个透明代理模型：在 COMPAS label 的 logistic loss
之外，加入一个 penalty，使真实 recidivists 中 Black 和 White 的平均预测分数更接近。

代理模型诊断：

- Black TPR：0.585
- White TPR：0.597
- TPR gap Black - White：-0.013
- 优化是否成功：True

这个代理模型只用于敏感性分析，不等同于原文的 Zafar 约束实现。

## 与原文的关键差异

1. 原文未公开代码、随机种子和完整参数。
2. 原文没有明确 logistic regression 的完整 feature matrix。
3. 原文没有明确 Figure 3A 的宽度是 raw transport mass，还是
   `mass * feature distance` 的 group-wise bias。
4. 原文没有明确 `Medium` COMPAS score 如何二分；本复现采用 `Low` vs
   `Medium/High`。
5. 原文没有明确是否先用全部 race 训练模型，再只画 Black/White，还是直接只用
   Black/White 子集。主报告采用 Black/White 子集。
6. 原文没有给出 equal opportunity classifier 的约束强度，导致 Figure 3B 无法
   精确复现。

结论：数据和 OT 框架可以复现，Section 4.2 的“跨 race/风险组的结构性偏差”
可以定性复现；但原文 Figure 3A/3B 的精确百分比无法只凭论文唯一推出。

## 输出文件

- `data/compas-scores-two-years.csv`：下载的 ProPublica 数据。
- `outputs/summary_metrics.csv`：所有分支的核心指标。
- `outputs/comparison_with_paper.csv`：原文数字与复现数字对比。
- `outputs/*_bias_row_pct.csv`：group-wise bias decomposition 矩阵。
- `outputs/*_mass_row_pct.csv`：raw transport mass decomposition 矩阵。
- `outputs/*_bias_heatmap.svg`：主要 bias decomposition 热力图。

## 复运行命令

```bash
/Users/zhangyongxiu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  section4_2_compas_reproduction/reproduce_compas_section4_2.py
```
