# IMPACT_AS 当前工具总结（不考虑是否接入 ASOT）

本文用于讨论 `IMPACT_AS` 当前版本到底已经是什么、主流程是什么、哪些能力是真正对外可用、哪些是代码里存在但没有挂到主入口的能力，以及后续适合讨论的优化方向。

## 1. 一句话定位

`IMPACT_AS` 目前本质上是一个基于 `PyQt5 + OpenCV` 的桌面视频标注工具，当前真正成型、并且从主入口可直接进入的两条主工作流是：

- `Action Segmentation`
- `Assembly State (PSR/ASR/ASD)`

从代码结构上看，`Action Segmentation` 是底层工作台，`Assembly State` 是建立在动作分割结果之上的上层状态推断与修订工作流。

如果先不讨论是否接入 ASOT，那么当前工具的核心不是“训练/推理平台”，而是“人工标注 + 半辅助修订 + 结果导入导出 + 数据清洗”的一体化桌面工作台。

## 2. 当前真正对外的主入口

主入口是 `app.py`，它启动 `MainWindow`。

当前 `MainWindow` 实际只开放两个任务：

- `Action Segmentation`
- `Assembly State (PSR/ASR/ASD)`

这意味着：

- `Action Segmentation` 和 `Assembly State` 是当前主产品面向用户的核心能力。
- 仓库里虽然还有 `HOIWindow`、HOI 快捷键、HOI 标注逻辑等代码，但它没有被当前 `MainWindow` 挂到主入口上。
- README 里提到的 `Transcript workspace support`、`HandOI / HOI annotation`，和当前主入口实际暴露的能力并不完全一致。

对讨论优化很重要的一点是：要区分“仓库里有代码”与“当前主产品真正可用的用户路径”。

## 3. 当前工具的整体产品形态

### 3.1 主体能力

当前版本可以概括为 4 层：

1. 视频加载与多视角同步查看
2. 动作分割标注与修订
3. 基于动作标签推导装配状态，并在状态时间线上修订
4. 围绕标注结果的导入、导出、日志、修复、批处理工具

### 3.2 它不是一个什么工具

当前公开版本不是一个完整的“端到端训练+推理平台”，也不是一个已经把所有研究模块产品化的统一平台。代码里还能看到一些未实现或未挂接的方向，例如：

- subtitle conversion
- bbox generator
- segmentation assistant
- interactive segmentation

这些在 `app.py` 末尾被明确标成未实现。

## 4. Action Segmentation 现在能做什么

`Action Segmentation` 是当前工具最完整的一块，也是其他工作流的基础。

### 4.1 基本会话能力

用户可以在一个会话里完成：

- 加载视频
- 为视频选择裁剪起止帧
- 通过 `Open Session...` 一次性加载视频、标签表和已有 JSON
- 导入/导出标注 JSON
- 导入/导出标签映射 TXT
- 保存操作日志和 validation summary

这部分能力主要集中在 `ui/action_window.py`。

### 4.2 多视角工作流

当前支持最多 5 个 view，并且要求各 view 的有效时间跨度一致。每个 view 有自己的：

- 视频路径
- 裁剪区间
- annotation store
- prelabel store
- entity/phase/anomaly store
- PSR 状态缓存

多视角之间支持：

- 同步播放
- 同步跳帧
- 统一时间轴对齐
- 按选中的 view 批量导入/导出 JSON
- 导出到“每个视角一个文件夹”的多视角结构

这说明当前工具已经不是单视频播放器，而是一个带 view-state 管理的标注工作台。

### 4.3 Coarse / Fine 两种动作标注模式

Action 标注支持两层粒度：

- `Coarse`：全局单轨动作标签
- `Fine`：按 entity 分轨标注

在 `Fine` 模式下，工具支持：

- entity 列表管理
- label 到 entity 的适用关系
- entity 可见性控制
- 每个 entity 独立 action store

也就是说，Fine 模式本质上已经是“多实体动作分割”。

### 4.4 Fine 模式下的 phase / anomaly 轴

在 Fine 模式下，还支持额外的阶段和异常信息：

- phase：`normal / anomaly / recovery`
- anomaly type：6 类异常向量

这让工具不只是记录动作段，还能在同一个片段上叠加：

- 动作类别
- 实体
- 阶段信息
- 异常类型

所以当前 action JSON 实际上已经兼顾了“动作 + 实体 + 阶段 + 异常”四层信息。

### 4.5 时间轴编辑能力

当前时间轴编辑是这个工具的核心交互能力之一，已经支持：

- segment 选中
- 拖动边界
- split
- delete
- snapping
- hover preview
- gap 提示
- view follow / pan
- 不同 row 的组合展示

Action 与 PSR 两条工作流都复用了同一套时间轴基础设施，说明 timeline 是整个系统的共用交互内核。

### 4.6 Manual Segmentation 和 Assisted Review

Action 工作流里已经有两套“半辅助”机制：

- `Manual Segmentation`
- `Assisted Review`

其中：

- Manual Segmentation 会记录人工切出来的 interaction spans
- Assisted Review 会基于候选边界和候选标签，逐点让用户确认

它不是全自动标注，而是“模型建议 + 人确认”的 review 机制。

### 4.7 预测结果导入和 prelabel baseline

当前工具已经支持导入外部模型输出：

- `txt` 三行式 segment 结果
- `json` segment 结果
- 可带 `top-k` 候选标签

导入后，工具会把模型结果存成：

- 当前人工标注 store
- 独立的 `prelabel_store`

这个设计很重要，因为它意味着当前工具已经把“模型预测”和“人工最终结果”做了结构性分离，不是简单覆盖。

### 4.8 Feature 和语义辅助能力

当前 Action 工具里已经接了可选的 feature/语义辅助链路，包括：

- 当前视频特征提取
- 外部特征导入
- I3D / ResNet50 feature 支持
- 基于 feature 的 boundary snap
- label prototype
- text bank
- SigLIP2 文本嵌入
- top-k 候选排序

这里的定位更像“辅助 review 的基础设施”，而不是一个完整推理系统。

### 4.9 Validation / Review 机制

当前工具支持：

- validation 模式切换
- validation overlay
- review queue
- review log 导入
- Accept / Reject 某条修改

这说明工具已经不只是“做标注”，也在服务“复核”和“标注质量追踪”。

### 4.10 Action 工作流的输出

当前 Action 侧主要可以输出：

- Native JSON
- 多视角按文件夹导出
- seed dataset
- label map TXT
- optional operation log CSV
- validation summary TXT

所以它已经具备相对完整的数据落盘链路。

## 5. Assembly State（PSR/ASR/ASD）现在能做什么

当前主入口的第二条主工作流是 `Assembly State (PSR/ASR/ASD)`。

它的本质不是独立从零标注，而是：

- 基于动作段
- 结合组件表和规则
- 推导组件状态事件
- 生成状态序列
- 允许人工在状态时间线上修订

### 5.1 当前工作方式

切换到 Assembly State 后：

- 左侧仍然是视频和共享时间轴
- 右侧变成 PSR 状态编辑面板
- ActionWindow 被重挂到 PSR 容器中

这意味着当前 PSR/ASR/ASD 不是完全独立的第二个程序，而是 Action 工作台的一种专用模式。

### 5.2 固定组件目录

当前最重要的现状之一是：

- PSR 侧使用固定的 HAS component catalog
- 即使加载外部 component 文件，也主要是做校验
- 实际推断仍然依赖内部固定组件表

也就是说，当前装配状态工作流并不是一个完全通用的“任意组件装配状态工具”，而是对当前 HAS 场景高度定制的。

### 5.3 规则驱动的状态推导

当前 Assembly State 的核心链路是：

1. 从动作标签读取 segment
2. 根据 rules 把动作 label 映射到 component-state event
3. 构造 state sequence / state runs
4. 在时间轴上显示每个组件或 combined state
5. 允许人工调整

所以当前 PSR/ASR/ASD 的“自动化”本质上是规则驱动推导，不是端到端模型推理。

### 5.4 状态编辑能力

当前状态编辑已经比较完整，支持：

- 选中当前 segment
- 从当前片段开始应用
- split at playhead
- reset segment
- invert segment
- merge adjacent identical segments
- no-gap timeline
- auto-carry next
- undo / redo

这套能力说明当前 Assembly State 页已经具备“可生产使用的修订面板”雏形。

### 5.5 trace analysis 和 conflict review

当前系统还做了 procedure trace 分析，能够识别类似问题：

- early commit
- missing commit

然后把这些问题转成：

- review item
- repair candidate
- review queue

这是当前版本里比较有价值的一点，因为它让状态推导不只是“出结果”，而是还能“发现值得人工复核的状态冲突”。

### 5.6 导入导出能力

当前 PSR/ASR/ASD 相关能力包括：

- load/save components
- load/edit/export rules
- load assembly-state JSON
- export assembly-state JSON
- batch convert action dataset to assembly-state JSON

但需要特别说明：

- 当前导出函数实际上只开放了 `ASR` 导出
- 虽然 UI 和命名写的是 `PSR/ASR/ASD`
- 但在当前公开实现里，真正启用的是 `ASR` 输出链路

这在产品命名和真实能力之间存在一定错位。

## 6. 当前 JSON / 数据结构的实际特点

### 6.1 Action 侧

Action 侧不是单一 store，而是多套 store 并存：

- 主 action store
- extra/manual segmentation store
- prelabel store
- per-entity action stores
- per-entity phase stores
- per-entity anomaly-type stores

这使得当前工具的表达能力很强，但也意味着状态同步复杂度比较高。

### 6.2 Native JSON 侧

Action/Fine JSON 里当前可表达的信息包括：

- video_id
- view 信息
- labels / action_labels
- segments
- entity
- phase
- anomaly_type
- verbs
- nouns
- meta_data

### 6.3 ASR 导出侧

当前 ASR 导出不只是导出 sparse event，而是同时导出：

- components
- initial_state / initial_state_vector
- state_sequence
- state_changes
- workflow
- meta_data
- procedure_trace
- repair_candidates

这说明当前导出已经带有“可被下游继续消费和调试”的信息密度。

## 7. 仓库结构上现在的真实形态

从实现上看，当前系统高度集中在少数几个核心文件中：

- `app.py`：入口
- `ui/main_window.py`：主任务切换
- `ui/action_window.py`：Action 主工作台，同时也承载大量 PSR 逻辑
- `ui/psr_window.py`：PSR 右侧状态面板
- `ui/timeline.py`：共用时间轴
- `core/models.py`：底层 annotation store
- `core/psr_state.py`：规则与状态推导
- `core/procedure_trace.py`：trace 分析

其中最关键的现实情况是：

- `ui/action_window.py` 约 `20085` 行
- `ui/hoi_window.py` 约 `5460` 行
- `ui/timeline.py` 约 `2984` 行

这说明当前系统的主要问题已经不在“功能有没有”，而在“功能和职责是否过度集中”。

## 8. 关于 HOI、Transcript、以及未挂接能力的现状

### 8.1 HOI

仓库中确实有 `ui/hoi_window.py`，且功能不少，包括：

- 单视频 HOI 标注
- YOLO 框读取/检测
- MediaPipe Hands
- verb 管理
- HOI timeline
- validation / undo / redo

但是它当前没有被主入口挂出来，因此：

- 它属于“仓库内存在的能力”
- 但不属于“当前主产品入口下可直接使用的标准工作流”

### 8.2 Transcript

README 提到 `Transcript workspace support`，但在当前公开代码主路径里，我没有看到一个明确、独立、已挂接的 transcript 专用窗口或主流程。

因此更稳妥的表述应该是：

- transcript 相关能力在公开版本中不是当前主线工作流
- 至少从当前主入口和主要 UI 结构来看，它不是一个清晰可用的独立模块

### 8.3 未实现能力

当前代码中还明确保留了若干未实现项，这意味着这个项目带有明显的“研究演化痕迹”，并非所有历史方向都已产品化。

## 9. 如果先不谈 ASOT，当前工具的核心结论

可以把当前 `IMPACT_AS` 视为：

- 一个以 `Action Segmentation` 为底座的桌面标注工作台
- 其上叠加了 `Fine entity / phase / anomaly` 表达能力
- 再叠加了 `Assembly State` 的规则推导、状态编辑和 trace review
- 同时具备多视角、导入导出、日志、validation、数据修复脚本等配套能力

所以当前版本的真实优势是：

- 工作流比较完整
- 人机协作链路已经具备雏形
- 数据导入导出和维护脚本较丰富

而不是：

- UI 非常轻量
- 模块边界已经清晰
- 所有 README 能力都已稳定对外

## 10. 关于 ASOT 的当前结论

如果你们接下来要讨论“要不要加入 ASOT”，我建议先基于一个事实：

- 当前主工作流里，我没有看到 ASOT 已经实质接入产品功能链路
- 仓库里与 `asot` 最直接相关的公开痕迹主要是 `runner_envs.json` 里的环境 profile

也就是说：

- 当前总结可以完全独立于 ASOT 成立
- 如果后续加入 ASOT，它更像是在现有 Action/Assisted Review 链路上扩展一个新的模型或辅助模块
- 而不是替代当前工具的主体架构

## 11. 当前最明显的优化讨论点

下面这些点，比较适合拿去和别人讨论“下一步怎么优化”。

### 11.1 先做产品边界收敛

建议先统一回答几个问题：

- 当前产品到底只保留 `Action + Assembly State`，还是要把 HOI 重新纳入主入口？
- README 里写的能力，哪些是“当前版本真的支持”，哪些只是历史遗留描述？
- `PSR/ASR/ASD` 这个命名是否需要和当前“实际上主要导出 ASR”保持一致？

如果产品边界不先收敛，后面做代码重构很容易继续扩散。

### 11.2 把 `ui/action_window.py` 拆层

这是当前最大的工程问题。

建议至少拆成几层：

- 纯 UI 组件层
- annotation session / view-state 管理层
- action annotation domain 层
- PSR domain 层
- import/export 层
- assisted review / feature / text-bank 层

只要这一层不拆，后续任何能力新增都会继续堆在单文件里。

### 11.3 明确“人工结果”和“模型建议”的统一接口

当前其实已经有雏形：

- manual store
- prelabel store
- assisted candidates
- review queue

下一步可以考虑把它们统一成清晰的数据合同，例如：

- baseline prediction
- accepted correction
- rejected suggestion
- locked segment
- repair candidate

这会让后续无论接 ASOT、别的时序模型、还是别的 boundary proposal，都能走同一条接口。

### 11.4 把 Assembly State 从“场景定制”提升到“框架化”

当前 Assembly State 很强，但明显高度绑定当前 HAS 组件目录。

如果后面要扩展，至少要讨论：

- 组件目录是否还固定
- rule schema 是否通用
- state label 和 workflow 是否配置化
- 不同 model profile 的差异是否要变成明确配置，而不是散在逻辑里

### 11.5 统一导入导出 schema

当前 Action JSON、Fine JSON、ASR JSON 都已经比较丰富，但也出现了：

- Native schema
- adapter schema
- legacy schema
- batch conversion schema

后续最好整理成“主 schema + adapter 层”，否则工具内部和工具脚本会继续分叉。

### 11.6 给关键工作流补自动化测试

当前仓库里基本没有自动化测试，这是一个实际风险点。

优先建议补的不是 UI 测试，而是：

- AnnotationStore 行为测试
- JSON load/save roundtrip
- half-open segment 修正测试
- PSR rule -> event -> state sequence 测试
- trace conflict 分析测试
- 多 view 导出测试

这些测试一旦补上，后续重构会安全很多。

### 11.7 重新整理 README 和文档

现在的 README 比较像“历史能力列表”，不完全等于当前主入口。

建议后续文档至少拆成：

- 当前公开版本真正支持的工作流
- 可选/实验性能力
- 仓库内保留但未挂接的模块
- 已弃用或未实现能力

这样外部讨论会更聚焦，也更容易管理预期。

## 12. 适合直接拿去讨论的一句总结

如果要非常压缩地描述当前工具，我建议可以直接这样说：

`IMPACT_AS` 当前版本是一个以 Action Segmentation 为底座、支持多视角、Fine entity/phase/anomaly 标注、并可进一步推导和修订 Assembly State 的桌面标注工作台；它已经具备比较完整的人工标注、半辅助 review、导入导出和数据维护能力，但工程结构仍然高度集中，产品边界和公开文档也还没有完全收敛。
