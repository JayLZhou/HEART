这个 BO 框架的核心思路可以概括成一句话：

**让 LLM 不直接替代 BO 选点，而是每一轮给 BO 一个“偏好的区域/点”，再把这个偏好稳定地注入 GP surrogate 里，最后仍由 acquisition function 负责真正选下一个实验点。**

结合论文 `4356_Unleashing_LLMs_in_Bayesi.pdf` 和代码 `exp.py`，大体流程是这样的：

1. 先用少量初始点建立一个普通 GP surrogate。  
在这个 toy 实验里，初始点是 Sobol 采样，不是 LLM 给的，见 `run_case(...)` 里的初始化逻辑。

2. 每一轮把“历史实验 + 上一轮 reasoning + 当前任务背景”发给 LLM。  
LLM 只能输出两种结构化建议之一：  
`[point, [...], confidence]` 或 `[region, [[lb...],[ub...]], confidence]`。  
对应调用入口是 `call_chat(...)`，提示词定义在 `prompt.py`。

3. 把 LLM 输出变成一个 “preference plan”。  
这一步主要由 `build_expert_input_from_parsed(...)` 和 `decide_preference_tilt_from_expert(...)` 完成。  
论文里的关键点是：**不是直接改 acquisition，而是把 LLM 偏好转成一个对 GP 均值的 lift / tilt。**  
也就是“更相信某个区域可能更优”，但 **协方差不变**，这样不会把 BO 的不确定性结构搞坏。

4. 再由 BO 在这个“被 LLM 轻推了一下”的 surrogate 上选点。  
实现入口是 `propose_points_from_plan(...)`。  
如果是 `region` 模式，就走 region tilt；  
如果是 `point` 模式，就给点附近一个 bump prior。  
最后还是用 qLogEI / qEI 这类 acquisition 真正选实验点。

5. 执行真实评估，更新数据，再进入下一轮。  
主循环在 `run_case(...)`。

这篇论文相对普通 BO 的本质区别是：

- 普通 BO：只信历史数据。
- LLAMBO 一类方法：LLM 多半只做 warm start 或候选点生成。
- 这篇 LGBO：**LLM 每轮持续提供“语义偏好”，而 BO 始终保留最终决策权。**

所以它想解决的是两个问题：

- 冷启动慢：早期数据很少，GP 很弱，LLM 的领域常识能给搜索一个方向。
- 高维难搜：LLM 不一定能精确说出最优点，但常常能说出“哪一片区域更值得试”。

论文里把这叫 **region-lifted preference**。直观理解就是：

- LLM 说“这片区域更可能好”
- 系统不把这话当硬约束
- 而是把 GP 的均值朝这片区域抬高一点
- 抬高幅度由 `confidence` 控制
- 于是 acquisition 会更愿意去那附近试，但仍能因为不确定性去探索别处

从代码角度，你可以把它理解成三层：

- `exp.py`：实验编排层，负责 LLM 对话、解析、循环更新
- `decide.py`：把 LLM 文本建议变成数学上的 preference/plan
- `boo.py`：把这个 plan 注入 BO 选点器

还有一个读代码时要注意的点：

**当前这个 `exp.py` 更像论文思想的 toy 版复现，不是完整论文系统。**  
比如论文正文强调 warm start + continuous guidance，但这里 toy case 的初始点实际上是 Sobol，不是 LLM 初始化；它保留的重点是“每轮持续用 LLM 偏好引导 BO”。
