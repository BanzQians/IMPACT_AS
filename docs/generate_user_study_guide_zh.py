#!/usr/bin/env python3
"""Generate IMPACT-Scribe Chinese User Study Guide PDF via weasyprint."""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "IMPACT_Scribe_用户研究指南.pdf"
ASSETS = Path(__file__).resolve().parent / "assets" / "quick_start"


def img_tag(filename: str, caption: str = "") -> str:
    p = ASSETS / filename
    if not p.exists():
        return f'<p class="caption">[图片缺失: {filename}]</p>'
    html = f'<img src="file://{p}" alt="{caption}">'
    if caption:
        html += f'\n<p class="caption">{caption}</p>'
    return html


HTML = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
@page {{
  size: A4;
  margin: 22mm 20mm 22mm 20mm;
  @bottom-center {{
    content: counter(page);
    font-size: 9pt;
    color: #888;
  }}
}}
body {{
  font-family: "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei",
               "PingFang SC", "Hiragino Sans GB", sans-serif;
  font-size: 11pt;
  line-height: 1.65;
  color: #222;
}}
h1 {{
  font-size: 26pt;
  color: #17212b;
  text-align: center;
  margin-top: 120px;
}}
h1.sub {{
  font-size: 18pt;
  color: #506580;
  margin-top: 8px;
}}
.title-meta {{
  text-align: center;
  color: #888;
  font-size: 11pt;
  margin-top: 60px;
}}
h2 {{
  font-size: 16pt;
  color: #17212b;
  border-bottom: 1px solid #d0d5da;
  padding-bottom: 4px;
  margin-top: 28px;
  page-break-after: avoid;
}}
h3 {{
  font-size: 13pt;
  color: #324860;
  margin-top: 18px;
  page-break-after: avoid;
}}
p {{ margin: 6px 0; }}
ul {{ margin: 4px 0 8px 18px; padding: 0; }}
li {{ margin: 3px 0; }}
img {{
  display: block;
  max-width: 100%;
  border: 1px solid #d7dce2;
  border-radius: 8px;
  margin: 8px auto;
}}
.caption {{
  text-align: center;
  font-size: 9pt;
  color: #888;
  margin: 2px 0 12px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0;
  font-size: 10.5pt;
}}
th, td {{
  border: 1px solid #d0d5da;
  padding: 5px 10px;
  text-align: left;
}}
th {{
  background: #f0f2f5;
  font-weight: bold;
}}
.page-break {{ page-break-before: always; }}
.tip {{
  background: #f0f7ff;
  border-left: 4px solid #3b82f6;
  padding: 10px 14px;
  margin: 10px 0;
  border-radius: 4px;
}}
</style>
</head>
<body>

<!-- 封面 -->
<h1>IMPACT-Scribe</h1>
<h1 class="sub">用户研究指南</h1>
<p style="text-align:center; color:#888; margin-top:20px;">
  交互式动作分割审查工具
</p>
<p class="title-meta">
  CVHCI &nbsp;|&nbsp; 卡尔斯鲁厄理工学院 (KIT)<br>
  版本 1.0 &nbsp;|&nbsp; 2026 年 4 月
</p>

<!-- 1. 概述 -->
<div class="page-break"></div>
<h2>1. 概述</h2>
<p>
  IMPACT-Scribe 是一款交互式动作分割审查工具。给定一段视频和一个初始的（机器生成的）动作分割结果，
  您需要通过以下方式审查并修正分割：
</p>
<ul>
  <li><b>直接标注</b>：在时间轴上选择标签、拖拽画框来创建动作片段</li>
  <li><b>智能引导</b>：跟随系统的 Query 建议（边界或标签问题）</li>
  <li><b>Scribble 精修</b>：在时间轴上画涂抹笔触来微调边界位置</li>
  <li><b>确认或拒绝</b>：接受或拒绝系统提出的修正建议</li>
</ul>
<p>
  您的目标是用尽可能少的操作，产出高质量的动作分割标注。
  工具会从您的每一次修正中学习：每次 Accept 或 Reject 操作都会更新内部模型，
  使后续建议更加准确。
</p>

<!-- 2. 快速开始 -->
<h2>2. 快速开始</h2>

<h3>第一步：加载视频或基线</h3>
<p>
  点击工具栏左侧的菜单按钮，选择「Open Session」。选择一个视频文件，
  可选地加载一个已有的标注 JSON 文件。系统会加载视频并在时间轴上显示基线分割。
</p>
{img_tag("step_01_load_baseline.png", "图 1：加载视频和基线分割")}

<h3>第二步：查看工作区</h3>
<p>
  加载完成后，左侧显示视频播放器，右侧显示动作时间轴。
  每个彩色色块代表一个动作标签。将鼠标悬停在时间轴上可以预览对应帧。
</p>
{img_tag("step_02_loaded_workspace.png", "图 2：视频播放器与时间轴")}

<h3>第三步：点击 Suggest Query</h3>
<p>
  点击工具栏中的「Suggest Query」按钮。系统会分析当前分割，
  建议下一个最值得审查的边界或标签问题。建议卡片显示在底部。
</p>
{img_tag("step_03_suggest_query.png", "图 3：Suggest Query 生成审查目标")}

<h3>第四步：审查建议</h3>
<p>
  底部卡片展示建议的问题：需要检查的边界或需要验证的标签。
  阅读描述后，决定是接受、拒绝还是用 Scribble 精修。
</p>
{img_tag("step_04_review_suggestion.png", "图 4：审查建议的边界或标签")}

<h3>第五步：精修并接受</h3>
<p>
  如果边界需要调整，在时间轴上可疑区域画一笔 uncertain scribble。
  系统会提出精修后的边界（红色标记线）。拖拽微调位置后点击 Accept。
</p>
{img_tag("step_05_refine_and_accept.png", "图 5：画 Scribble、精修、接受")}

<!-- 3. 核心操作流程 -->
<div class="page-break"></div>
<h2>3. 核心操作流程</h2>

<h3>3.1 直接在时间轴上标注（基础操作）</h3>
<p>
  最简单的标注方式是直接在时间轴上画框创建片段。无需切换任何特殊模式，随时可用：
</p>
<ol>
  <li><b>选择标签</b>：在左侧的标签面板中，先点击一个动词（Verb），再点击对应的物体（Object）</li>
  <li><b>创建片段</b>：在时间轴的空白区域，鼠标左键按住并拖拽，即可画出一个新的彩色片段</li>
  <li><b>调整边界</b>：拖拽已有片段的左边缘或右边缘来调整起止位置</li>
  <li><b>移动片段</b>：拖拽已有片段的中间区域来整体移动</li>
  <li><b>删除片段</b>：在片段上<b>右键点击</b>即可删除</li>
  <li><b>分割片段</b>：按住 <b>Ctrl + 鼠标左键</b>点击片段，在该帧处一分为二</li>
</ol>
<div class="tip">
  💡 这是最直觉的操作方式。当你需要从零标注或快速修复某个区域时，直接在时间轴上操作即可。
</div>

<h3>3.2 智能查询引导流程（Query Suggestion）</h3>
<p>
  为了高效审查，系统内置了查询规划器。它根据以下指标选择下一个最值得关注的问题：
</p>
<ul>
  <li><b>边界不确定性</b>：模型最不自信的位置</li>
  <li><b>标签分歧</b>：特征提示可能是不同标签的位置</li>
  <li><b>多视角冲突</b>：不同摄像头视角之间有分歧的位置</li>
  <li><b>状态冲突</b>：动作标签违反组装规则的位置</li>
</ul>
<p>点击 Suggest Query 后，有三个操作可选：</p>
<ul>
  <li><b>Accept Suggestion</b>（接受建议）—— 直接应用系统的修改方案</li>
  <li><b>Start Scribble</b>（开始涂抹）—— 进入 Scribble 模式手动精修边界</li>
  <li><b>Reject Suggestion</b>（拒绝建议）—— 跳过此问题，继续下一个</li>
</ul>
<p>
  每次操作后再次点击 Suggest Query，系统会根据你的修正自动调整后续建议。
</p>

<h3>3.3 时间涂抹交互（Temporal Scribble）</h3>
<p>
  Scribble 是一种在时间轴上画短笔触的方式，用于告诉系统某个位置可能存在边界。
  有三种涂抹类型：
</p>
<ul>
  <li><b>Uncertain（默认）</b>—— "我觉得这个范围内某处有一个边界"</li>
  <li><b>Left</b> —— "这个区域的左侧属于这个标签"</li>
  <li><b>Right</b> —— "这个区域的右侧属于这个标签"</li>
</ul>
<p><b>Scribble 操作流程：</b></p>
<ol>
  <li>通过 Interaction 下拉菜单选择「Boundary Scribble」或点击「Start Scribble」按钮</li>
  <li>在时间轴上，对可疑的边界区域按住鼠标左键拖拽画一笔</li>
  <li>系统自动提出一个边界分割点（红色标记），并标注左右两侧的标签</li>
  <li>可以拖拽红色标记来微调精确位置</li>
  <li>点击 Accept 应用修改，或 Reject 放弃</li>
  <li>点击「Clear Scribbles」清除当前涂抹</li>
</ol>

<h3>3.4 手动全局分割模式</h3>
<p>
  在 Interaction 下拉菜单中选择「Manual Segmentation」。
  此模式下可以逐帧精确放置边界。适合不需要智能辅助、需要完全手动控制的场景。
</p>

<!-- 4. 界面参考 -->
<div class="page-break"></div>
<h2>4. 界面参考</h2>

<h3>4.1 工具栏按钮</h3>
<table>
  <tr><th>按钮</th><th>功能</th></tr>
  <tr><td>▶ Play / Pause</td><td>播放/暂停视频</td></tr>
  <tr><td>◀◀ / ▶▶</td><td>前进/后退 10 帧</td></tr>
  <tr><td>ASOT Pre-label</td><td>运行模型生成基线分割</td></tr>
  <tr><td>Magnifier</td><td>切换视频放大镜</td></tr>
  <tr><td>Validation</td><td>切换验证覆盖层</td></tr>
  <tr><td>Interaction 下拉</td><td>Boundary Scribble / Manual Segmentation / Exit</td></tr>
  <tr><td>Clear Scribbles</td><td>清除当前涂抹笔触和建议</td></tr>
  <tr><td>Suggest Query</td><td>让规划器推荐下一个审查目标</td></tr>
  <tr><td>Settings (⚙)</td><td>打开设置（快捷键 Ctrl+,）</td></tr>
  <tr><td>Quick Start (ℹ)</td><td>打开内置快速入门指南</td></tr>
  <tr><td>+ Add View</td><td>添加额外的摄像头视角</td></tr>
</table>

<h3>4.2 时间轴</h3>
<p>时间轴是主要的标注区域。视觉元素说明：</p>
<ul>
  <li>彩色色块 = 动作片段（显示标签名称）</li>
  <li>红色竖线 = 当前播放头位置</li>
  <li>鼠标悬停 = 在视频播放器中预览该帧</li>
  <li>虚线 = 片段边界（可拖拽）</li>
  <li>红色标记 = Scribble 精修提出的边界建议</li>
  <li>滚轮 = 左右平移时间轴</li>
</ul>
<p><b>鼠标操作总结：</b></p>
<table>
  <tr><th>操作</th><th>效果</th></tr>
  <tr><td>左键拖拽空白区域</td><td>创建新片段（使用当前选中的标签）</td></tr>
  <tr><td>左键拖拽片段边缘</td><td>调整片段边界</td></tr>
  <tr><td>左键拖拽片段中间</td><td>整体移动片段</td></tr>
  <tr><td>右键点击片段</td><td>删除该片段</td></tr>
  <tr><td>Ctrl + 左键点击片段</td><td>在该帧处分割片段</td></tr>
</table>

<h3>4.3 标签面板</h3>
<p>
  左侧的标签面板按动词-物体结构组织所有可用标签。
  先点击动词筛选，再点击物体选中标签用于标注。
</p>
<ul>
  <li>搜索框：按名称过滤标签</li>
  <li>Add 按钮：创建新标签（名称、ID、颜色）</li>
  <li>双击物体列：重命名已有标签</li>
  <li>保留标签：「Unknown」「Other」「Background」始终显示在标签面板顶部，用于标记您不确定的片段</li>
</ul>

<h3>4.4 底部建议卡片（Footer）</h3>
<p>当查询建议激活时，底部显示：</p>
<ul>
  <li>查询类型：BOUNDARY SUGGESTION 或 LABEL SUGGESTION</li>
  <li>目标帧范围和涉及的标签</li>
  <li>置信度分数</li>
  <li>操作按钮：Accept / Reject / Start Scribble</li>
</ul>

<!-- 5. 快捷键 -->
<div class="page-break"></div>
<h2>5. 快捷键</h2>

<h3>导航</h3>
<table>
  <tr><th>快捷键</th><th>功能</th></tr>
  <tr><td>Space</td><td>播放 / 暂停</td></tr>
  <tr><td>A</td><td>后退 1 帧</td></tr>
  <tr><td>D</td><td>前进 1 帧</td></tr>
  <tr><td>Shift+A</td><td>后退 10 帧</td></tr>
  <tr><td>Shift+D</td><td>前进 10 帧</td></tr>
  <tr><td>J</td><td>后退 1 秒</td></tr>
  <tr><td>K</td><td>暂停</td></tr>
  <tr><td>L</td><td>前进 1 秒</td></tr>
  <tr><td>Home</td><td>跳到开头</td></tr>
  <tr><td>End</td><td>跳到末尾</td></tr>
</table>

<h3>编辑</h3>
<table>
  <tr><th>快捷键</th><th>功能</th></tr>
  <tr><td>Ctrl+Z</td><td>撤销</td></tr>
  <tr><td>Ctrl+Y</td><td>重做</td></tr>
  <tr><td>Ctrl+,</td><td>打开设置</td></tr>
</table>

<h3>组装状态模式 (PSR)</h3>
<table>
  <tr><th>快捷键</th><th>功能</th></tr>
  <tr><td>Ctrl+K</td><td>在播放头位置分割片段</td></tr>
  <tr><td>Ctrl+Shift+S</td><td>范围锁定到当前片段</td></tr>
  <tr><td>Ctrl+Shift+F</td><td>范围锁定从当前帧往后</td></tr>
  <tr><td>Ctrl+Backspace</td><td>重置选中片段</td></tr>
  <tr><td>Ctrl+I</td><td>反转选中片段状态</td></tr>
  <tr><td>Ctrl+M</td><td>合并相邻相同状态</td></tr>
</table>

<h3>视频播放器</h3>
<table>
  <tr><th>操作</th><th>功能</th></tr>
  <tr><td>Ctrl+滚轮</td><td>缩放视频</td></tr>
  <tr><td>左键拖拽</td><td>平移缩放后的视频</td></tr>
  <tr><td>双击</td><td>重置缩放为适应窗口</td></tr>
</table>

<!-- 6. 多视角 -->
<h2>6. 多视角支持</h2>
<p>
  IMPACT-Scribe 支持最多 5 个同步的摄像头视角。
  点击「+ Add View」加载额外视频。所有视角共享同一条时间轴，帧级同步。
</p>
<ul>
  <li>在任一视角中悬停或跳转，所有视角同步更新</li>
  <li>不同视角的标签可以不同（每个视角独立标注）</li>
  <li>查询规划器会考虑跨视角不一致性</li>
</ul>

<!-- 7. 导入导出 -->
<h2>7. 导入 / 导出</h2>
<table>
  <tr><th>功能</th><th>说明</th></tr>
  <tr><td>Open Session</td><td>加载视频 + 可选的标签映射 + 标注 JSON</td></tr>
  <tr><td>Import JSON</td><td>为选中的视角加载标注数据</td></tr>
  <tr><td>Export JSON</td><td>导出所有标注到单个 JSON 文件</td></tr>
  <tr><td>Export per-view</td><td>按视角分别导出 JSON</td></tr>
  <tr><td>Import/Export Label Map</td><td>加载或保存标签词汇表（TXT 格式）</td></tr>
</table>

<!-- 8. 用户研究任务说明 -->
<div class="page-break"></div>
<h2>8. 用户研究任务说明</h2>

<h3>8.1 您的任务</h3>
<p>
  系统会为您提供一段带有机器预生成动作分割的视频。
  您的任务是使用本工具的引导流程审查并修正这些分割标注。
</p>

<h3>8.2 操作流程</h3>
<ol>
  <li>系统会为您加载一个预配置的标注会话</li>
  <li>点击「Suggest Query」接收第一个审查问题</li>
  <li>对每个建议做出决定：接受 (Accept)、拒绝 (Reject)、或用 Scribble 精修</li>
  <li>每次决定后再次点击「Suggest Query」</li>
  <li>您也可以随时直接在时间轴上画框来创建、修改或删除片段</li>
  <li>当你对分割质量满意时，导出最终标注</li>
</ol>

<h3>8.3 使用建议</h3>
<div class="tip">
  <ul style="margin:0;">
    <li>随时可以直接在时间轴上拖拽画框来标注，这是最基础的操作</li>
    <li>信任查询规划器：它会优先推荐最有影响力的修正位置</li>
    <li>边界大致正确但不精确时，使用 Scribble 微调</li>
    <li>建议正确就直接 Accept，不确定就 Reject 跳过</li>
    <li>悬停在时间轴上可以快速预览对应帧的画面</li>
    <li>用 A/D 键逐帧查看，用于精确判断边界位置</li>
    <li>操作失误时 Ctrl+Z 撤销</li>
    <li>对于无法确定的片段，使用「Unknown」或「Other」标签标记</li>
  </ul>
</div>

<h3>8.4 什么时候算"完成"</h3>
<p>
  没有固定的修正次数要求。当您认为分割结果准确地反映了视频中的动作时，即可完成。
  系统会记录您的所有交互操作用于后续分析。
</p>

<!-- 9. 常见问题 -->
<h2>9. 常见问题</h2>
<table>
  <tr><th>问题</th><th>解决方法</th></tr>
  <tr><td>没有建议可用</td><td>确保已加载视频和基线标注</td></tr>
  <tr><td>Scribble 无法使用</td><td>确认已从 Interaction 下拉菜单切换到「Boundary Scribble」模式</td></tr>
  <tr><td>视频无法播放</td><td>检查视频文件路径是否可访问，尝试重新加载</td></tr>
  <tr><td>标签缺失</td><td>通过菜单导入标签映射 TXT 文件</td></tr>
  <tr><td>需要帮助</td><td>点击右上角 Quick Start 按钮查看内置引导</td></tr>
</table>

</body>
</html>
"""

if __name__ == "__main__":
    from weasyprint import HTML as WPHTML
    WPHTML(string=HTML).write_pdf(str(OUT))
    print(f"Generated: {OUT}")
