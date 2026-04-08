# IMPACT-Scribe 人类测试操作说明

这份说明只定义当前论文主线需要的三种对比条件，避免把参与者带入过多无关功能。

## 目标

比较以下三种标注协议在相同视频、相同起始条件下的效率和最终质量：

1. 其他现有工具的手动标注
2. IMPACT-Scribe 的 `Scribble Only`
3. IMPACT-Scribe 的 `Scribble + Planner`

## 公平性原则

- 三种条件尽量使用同一批视频。
- 如果使用初始粗分割或 prelabels，三种条件都必须使用同一份起始结果。
- 如果外部手动工具无法导入 prelabels，应明确记录该条件是 `from scratch`，不要和 `with prelabels` 混用。
- 每位参与者都应先完成简短练习，再进入正式计时。

## 条件 A：其他工具手动标注

这部分不在 IMPACT-Scribe 内完成。

建议记录：

- 开始时间、结束时间
- 使用的视频 ID
- 是否使用初始 prelabels
- 最终导出的标注文件路径

建议导出的指标：

- `Boundary-F1@10 / @25 / @50`
- `edit score`
- `frame accuracy`
- 完成时间

## 条件 B：Scribble Only

### 参与者操作

1. 打开 `app.py`
2. 载入视频、特征、已有 annotation / prelabel
3. 在顶部 `Study:` 下拉框中选择 `Study: Scribble Only`
4. 直接在时间轴上画 boundary scribble
5. 如果 proposal 不满意，可以在同一处继续补画，或直接拖动 proposal 边界
6. 点击 `Accept Boundary`
7. 重复直到完成当前视频

### 这一条件下系统行为

- study 模式固定为 `Coarse`
- `Fine` 相关入口隐藏
- `Interaction` 下拉隐藏
- planner 关闭
- 参与者需要自己决定下一处边界在哪里

## 条件 C：Scribble + Planner

### 参与者操作

1. 打开 `app.py`
2. 载入视频、特征、已有 annotation / prelabel
3. 在顶部 `Study:` 下拉框中选择 `Study: Scribble + Planner`
4. 点击顶部 `Next Boundary`
5. 系统会自动跳到下一处建议边界
6. 在时间轴上画 boundary scribble；必要时补画或拖动 proposal 边界
7. 点击 `Accept Boundary`
8. 再次点击 `Next Boundary`，或按当前界面流程继续到下一处建议边界
9. 重复直到完成当前视频

### 这一条件下系统行为

- study 模式固定为 `Coarse`
- `Fine` 相关入口隐藏
- `Interaction` 下拉隐藏
- planner 只推荐 `boundary`，不混入 label review 或 state repair

## 当前 Study 模式下的边界语义

当前论文主线只使用统一的 `boundary scribble` 手势：

- 画在空白区域：表示新建边界
- 在已有 boundary 上画窄而短的 scribble：表示删除该边界并 merge 相邻段
- 在已有 boundary 附近画更宽、跨越边界的 scribble：表示细化该边界

所有这些都复用同一个 `Accept Boundary` 流程。

## Blank Canvas 语义

在空白画布下，系统按连续时间分段工作，而不是局部补丁：

- 第一个接受的 boundary 会填满它前面紧邻的无标注时间段
- 第二个接受的 boundary 会继续填满下一个紧邻的无标注时间段
- 已经走过的时间轴不应该留下 gap

换句话说，blank canvas 下应当表现为沿时间轴逐段向前切分和填充。

## 选中与绘制

为减少误操作，当前交互优先级是：

- 单击已有 segment / marker：优先选中
- 只有拖动超过阈值时，才真正开始 scribble
- 右键保留删除已有标注或 marker 的语义

因此，参与者不应先在很多位置都画完再统一接受，而应一次只处理一个边界：

1. 画或调整当前边界
2. 查看 proposal
3. 点击 `Accept Boundary`
4. 再进入下一个边界

## 标签辅助

当前论文主线仍然是 boundary-first，但 label assistance 已经接入当前流程：

- 左侧面板会显示当前 segment / gap 的推荐标签
- 对于 `blank fill` 和 `remove boundary` proposal，可以在 `Accept Boundary` 之前先点选推荐标签进行覆盖
- 候选标签来自 prototype memory、导入候选、可选 text prior 和 runtime confusion memory

注意：

- study 模式下左侧面板标题会显示为 `Recommended Labels` / `Labels`
- 这条标签路径是辅助，不是参与者必须首先掌握的主流程

## IMPACT-Scribe 内部入口

当前用于人类测试的主入口只有这些：

- `Study:` 下拉框  
  用于切换 `Standard / Scribble Only / Scribble + Planner`

- `Next Boundary`  
  只在 `Scribble + Planner` 条件下使用，用于跳到下一处建议边界

- 时间轴上的 `Boundary Scribble`

- footer 中的 `Accept Boundary`

## 建议记录字段

建议每个 session 记录：

- `participant_id`
- `condition`
- `video_id`
- `start_time`
- `end_time`
- `total_elapsed_seconds`
- `interaction_count`
- `accept_count`
- `reject_count`
- `second_correction_count`
- `output_annotation_path`
- `output_scribble_sidecar_path`

IMPACT-Scribe 会在 correction session 和 scribble sidecar 中记录 `study_condition`，便于后续按条件分组。

## 推荐比较方式

推荐两条主比较：

1. `其他工具手动标注` vs `Scribble Only`  
   比较 boundary scribble 交互本身是否更省力。

2. `Scribble Only` vs `Scribble + Planner`  
   比较 planner 是否能减少搜索成本，并提高单位时间收益。

不要把 `其他工具手动标注` 直接和 `Scribble + Planner` 当成唯一对照；中间必须保留 `Scribble Only`，否则无法拆出 planner 的贡献。
