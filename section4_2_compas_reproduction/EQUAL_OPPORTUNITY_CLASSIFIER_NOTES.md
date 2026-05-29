# Equal Opportunity Classifier 推测说明

## 论文内线索

Section 4.2 说第三个 logistic regression 训练在 COMPAS labels 上，并“employing an equal opportunity constraint based on Zafar et al. [2017]”。论文背景部分把 Equal Opportunity 定义为：

```text
E[h(X) | Y = 1, A = 1] = E[h(X) | Y = 1, A = 0]
```

也就是不同敏感组在真实正类 `Y=1` 条件下的预测正率相等。对于二分类器，这等价于 equal TPR，或 equal FNR。

## 互联网资料线索

Zafar 团队公开了 `fair-classification` 仓库：

- 仓库 README/代码入口：https://github.com/mbilalzafar/fair-classification
- Disparate mistreatment README：https://github.com/mbilalzafar/fair-classification/tree/master/disparate_mistreatment
- COMPAS demo：https://github.com/mbilalzafar/fair-classification/tree/master/disparate_mistreatment/propublica_compas_data_demo

这个仓库把约束分成：

- `cons_type = 1`：约束 false positive rate。
- `cons_type = 2`：约束 false negative rate。
- `cons_type = 4`：同时约束 FPR 和 FNR。

因为 Equal Opportunity 是 equal TPR，而 `TPR = 1 - FNR`，所以最贴近本文 Figure 3B 的 Zafar-style classifier 应该是 **disparate mistreatment 里的 FNR constraint (`cons_type = 2`)**，或者某种等价的 equal-TPR 约束。Zafar README 的 COMPAS demo 默认展示的是 FPR 约束，但那是 demo 选择；本文明确写的是 Equal Opportunity，因此我判断 Figure 3B 更可能使用 FNR/TPR 约束。

## 本复现采用的代理模型

原文没有给出以下信息：

1. Zafar 代码版本。
2. 训练/测试划分。
3. 约束类型是否严格为 FNR，或是否同时约束 FPR/FNR。
4. `tau`、`mu`、covariance threshold 等超参数。
5. logistic regression 的完整 feature matrix。

因此本项目没有把代理模型伪装成精确复现。脚本实现的是一个透明的 equal-opportunity proxy：

```text
logistic loss on COMPAS label
+ penalty * (mean_score_black_true_positive - mean_score_white_true_positive)^2
```

它的目标是让真实 recidivists 中 Black 和 White 的预测分数更接近，从而模拟 equal TPR/FNR 约束的方向。生成 Figure 3B 风格图时，右侧 target group 由该 proxy classifier 的 `score >= 0.5` 得到。

## 如何理解 A/B 图

新生成的图：

- `outputs/figure3A_compas_style.svg`
- `outputs/figure3B_equal_opportunity_proxy_style.svg`
- `outputs/figure3_ab_style.svg`

这些图用于复现论文 Figure 3A/3B 的视觉结构和实验逻辑：

- 左侧是 `Ground Truth`，按真实 two-year recidivism 分成 Low/High。
- 右侧是 `Predicted`，A 图用 COMPAS 二元风险标签，B 图用 equal-opportunity proxy classifier。
- 四个组为 `WhiteLow`、`WhiteHigh`、`BlackLow`、`BlackHigh`。

注意：图的分支宽度不是论文原始宽度，不能作为精确数值复现使用。
