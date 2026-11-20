# 已完成的本地评估摘要

这里只保存完成运行的 `metrics.json` 和 `config.json`，保留原始数字和环境记录。config 中 Windows 路径用于记录来源，服务器运行时应传入服务器实际路径。

| 目录 | 数据范围 | 用途 |
| --- | --- | --- |
| `ic_pcs_test_20261119` | IC test2017，64 图 / 832 实例 | 每图随机一个实例框的完整自定义 baseline，非论文 COCO 协议 |
| `coco_smoke_3_images_20261120` | 本地 COCO 子集中的 3 图 / 3 查询 | 单框 AP⁺ 脚本功能检查，不代表完整 COCO 结果 |
| `coco_smoke_4_categories_20261120` | 1 图 / 4 类别查询 | 多类别提示隔离与 AP⁺ 流程检查，不代表完整 COCO 结果 |

IC 结果：segm AP 0.40034823385536655，bbox AP 0.4875458119004998；foreground_union 的 macro IoU 0.5651767219858903、macro Dice 0.6776738995937738。

不收录尚未完成的 COCO 328 图运行，不将小规模验证数值当作论文复现结果。原始预测、逐查询日志、工作簿暂存、权重和数据集留在本机，`outputs/` 已在 Git 忽略规则中。
