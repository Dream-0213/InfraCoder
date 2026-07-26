# InfraCoder 搭建过程逐文件讲解

以下是我搭建 InfraCoder 时每个文件的实现思路，按实际开发顺序排列。你可以顺着这个顺序读，每读完一个文件就知道下一步为什么需要下一个文件。

---

## 0. 先搞清楚需要什么

动手之前我想清楚了三件事：

- **核心机制**：Agent 的本质是一个循环——用户输入 → 调 LLM → LLM 选择返回文本或调用工具 → 执行工具 → 结果回填 → 继续循环，直到 LLM 主动返回文本
- **必须兼容私有化部署**：我需要同时支持 OpenAI 的标准 API 和一个叫 LiteLLM 的库（它可以对接 100 多种模型，包括 Anthropic、AWS Bedrock、Google Vertex 这些非 OpenAI 兼容的服务）
- **工具要可扩展**：我希望加一个新工具就是新建一个 Python 类，不需要改其他代码

想清楚这三点后，我开始写代码。从最底层的模块往上搭。

---

## 1. config.py — 配置管理（第一个文件）

这个文件最独立，不和任何模块耦合，所以我从它开始。

**我要解决什么问题**：项目启动时要从环境变量读取 API Key、模型名、Base URL 等配置。不同公司用的环境变量名不一样——有人用 `OPENAI_API_KEY`，有人用 `DEEPSEEK_API_KEY`，还有人可能之前用过另一个叫 CoreCoder 的项目，他们的环境变量是 `CORECODER_API_KEY`。

**怎么实现的**：我定义了一个 `Config` 数据类，里面有几个字段：model、api_key、base_url、max_tokens、temperature、max_context_tokens、provider。然后写了一个 `from_env()` 类方法，从环境变量里读这些配置，支持 `INFRACODER_*`、`CORECODER_*`、`OPENAI_*` 三种前缀环境变量，按优先级从上到下 fallback，只要找到一个就不往下找了。

**踩的一个坑**：`.env` 文件可能不在项目根目录——用户可能在子目录里启动项目。所以我写了一个 `_load_dotenv` 函数，从当前目录开始向上遍历目录树，直到找到 `.env` 文件或者到 home 目录为止。这样不管用户在哪个目录启动，配置都能自动加载。

---

## 2. llm.py — LLM 接口封装

有了配置，接下来我需要一个能和 LLM 通信的模块。

**我要解决什么问题**：不同 LLM 提供商返回的数据格式有差异——OpenAI 的流式 chunk 结构、DeepSeek 额外的 `reasoning_content` 字段、LiteLLM 的异常类型。我不想在 Agent 主循环里处理这些差异，所以需要一层统一的封装。

**怎么实现的**：先定义了一个 `LLMResponse` 数据类，包含 content（文本回复）、tool_calls（工具调用列表）、prompt_tokens 和 completion_tokens。不论底层是什么模型，最终都返回这个统一结构。

然后写了 `LLM` 类——它直接使用 OpenAI 的 Python SDK。初始化时创建一个 `OpenAI` 客户端，保存 model 名和其他参数。核心方法是 `chat()`，接收消息列表和可选的 tools 定义，返回一个 `LLMResponse`。

**流式处理的关键**：模型是流式返回的，工具调用的参数可能跨多个 chunk 传输。我维护了一个 `tc_map` 字典，key 是 tool call 的 index，value 累积它的 id、name 和 arguments（arguments 是字符串拼接的），流结束后用 `json.loads` 解析 arguments。

**重试策略**：写了 `_call_with_retry` 方法——429 限流和 5xx 服务端错误用指数退避重试最多 3 次，4xx 客户端错误直接抛出。有个细节：OpenAI 的 APIError 不是所有版本都有 `status_code` 属性，所以用了 `getattr(e, "status_code", None)` 做防御性读取。

**费用估算**：内置了一个 `_PRICING` 字典，包含 OpenAI、DeepSeek、Claude、Qwen、Kimi 的每百万 token 定价。每次请求记录 token 用量，`estimated_cost` 属性返回累计花的钱。模型名不在字典里就返回 None，不算错。

**LiteLLM 后端**：我写了 `LiteLLM` 类继承 `LLM`——它不走 OpenAI 的 `__init__`（不创建 OpenAI 客户端），而是通过 `litellm.completion()` 来调用。LiteLLM 本身会处理不同提供商的差异。它的重试逻辑用字符串匹配判断是否是临时错误（检查 `"rate_limit"`、`"timeout"`、`"connection"`、`"502"` 等关键词在异常信息里）。

**一个兼容性处理**：OpenAI 的 `stream_options` 参数在 vLLM 等自建服务上可能不支持。我先带上这个参数请求，如果遇到 `BadRequestError`（400 错误）就去掉参数重试，如果是其他异常就正常走重试逻辑。

---

## 3. tools/base.py — 工具基类

LLM 层写好了，现在需要让模型能做事情——这就需要工具系统。

**我要解决什么问题**：我需要一个统一的工具接口——LLM 通过名字调用工具、传入参数、拿到文本结果。加一个新工具不应该改现有代码。

**怎么实现的**：一个简单的抽象类，四个要素：

```python
class Tool(ABC):
    name: str           # 工具名，LLM 通过这个名字调用
    description: str    # 描述，LLM 根据这个决定什么时候用这个工具
    parameters: dict    # JSON Schema，告诉 LLM 这个工具有哪些参数
    execute(**kwargs)   # 实际执行逻辑，返回字符串结果
    schema()            # 转成 OpenAI function-calling 格式
```

每个具体的工具继承 `Tool`，实现 `execute()`。`schema()` 方法通用，返回 OpenAI 的 function schema 格式。工具注册表就是一个列表（在 `tools/__init__.py` 里），加新工具就是 `cls = NewTool()` 然后 `ALL_TOOLS.append(cls)`。

**为什么这么简单**：我不想引入复杂的设计模式。工具系统不需要做到百级扩展——私有化部署场景最多十几个工具，够用就行。

---

## 4. 具体工具——从最简单的开始

我有 base 了，开始写具体工具。我写顺序是：read → write → edit → grep → glob → bash → agent → gpu_status → vllm_status，从简单到复杂。

### read.py — 文件读取（最简单的工具）

**实现**：接收 file_path、可选的 offset（从第几行开始，1-based）和 limit（读多少行，默认 2000）。读取文件后按行分割，从 offset 开始取 limit 行，每行前面加上行号。如果文件超过 limit，末尾加一条 `... (total lines total, showing start-end)`。

**一个处理**：UTF-8 解码失败时用 `errors="replace"` 容错，不抛异常。

### write.py — 文件写入

**实现**：接收 file_path 和 content。用 `Path.expanduser().resolve()` 处理路径中的 `~/`，然后 `parent.mkdir(parents=True, exist_ok=True)` 自动创建父目录。写入后记录文件路径到一个全局集合 `_changed_files`，方便后面 `/diff` 命令查看本次修改了哪些文件。

**一个设计决策**：write_file 只用于创建新文件或完整重写，小改动应该用 edit_file。

### edit.py — 精确字符串替换编辑（核心工具）

这是我花功夫最多的工具。

**思路**：Claude Code 用的是一个叫"精确字符串匹配替换"的方式——LLM 告诉我要把什么内容替换成什么内容，如果 old_string 在文件中恰好出现一次就执行，否则报错。这样保证了编辑的安全性——不会因为匹配到错误的地方而修改了不该修改的代码。

**实现**：接收 file_path、old_string、new_string。读文件后 `content.count(old_string)` 查出现次数：

- 0 次：返回错误 + 文件前 500 字符预览，方便 LLM 看实际内容
- 多次：返回错误提示不够独特，需要加更多上下文
- 1 次：执行替换，生成 unified diff 返回

为什么返回 diff？因为 LLM 需要知道它改对了，后续决策也依赖这个信息。

**一个边界处理**：如果文件不是 UTF-8 编码（比如二进制文件或 latin-1），返回明确错误而不是抛出乱码。

### grep.py — 文本搜索

**实现**：接收 pattern（正则）、可选 path（搜索范围）和 include（文件类型过滤）。编译正则，遍历文件逐行匹配，返回 `文件路径:行号:内容` 格式的结果。上限 200 条匹配，防止返回太多塞爆上下文。

**一个细节**：搜索时会跳过 `.git`、`node_modules`、`__pycache__` 等目录。跳过的逻辑不是简单的"路径名包含就跳过"——如果搜索根目录本身叫 `build`（比如用户在 `/build/proj/` 里搜索），跳过规则只检查 `relative_to(root)` 的部分，不会把祖先目录的 `build` 也算进去把整个搜索树隐藏了。

### glob.py — 文件名搜索

**实现**：接收 pattern（glob 模式）和可选 path。用 Python 的 `Path.glob()` 搜索，结果按修改时间倒序排列（最新的文件优先展示），最多返回 100 条。

### bash.py — Shell 命令执行（最敏感的工具）

**实现**：接收 command 和可选的 timeout（默认 120 秒）。用 `subprocess.run` 执行命令，捕获 stdout 和 stderr。

**安全机制**：我列了 12 种危险命令的正则模式，包括：

- 递归删除根目录：`rm -rf /`、`rm -fr /`、`rm --recursive --force /`（各种 flag 顺序都覆盖了）
- 格式化文件系统：`mkfs`
- 原始磁盘写入：`dd if=... of=/dev/`
- 覆盖块设备：`> /dev/sd*`
- 修改根目录权限：`chmod 777 /`
- Fork 炸弹：`fork()`
- 管道下载执行：`curl ... | bash`、`wget ... | sh`

触发任何一条就返回 `Blocked:` 提示，不会执行。

**输出截断**：超过 15000 字符的输出会被截断为 6000 + `... truncated ...` + 3000 的格式，保留头尾关键信息。

**一个有意思的设计**：我做了跨命令的 `cd` 追踪。因为 bash 命令是每次新建进程执行的，`cd` 不会影响下一次的运行环境。所以我用一个线程局部变量 `_local.cwd` 来追踪当前工作目录。每次执行完命令后，检查命令字符串里是否有 `cd`，如果有就解析目标路径，更新 `_local.cwd`。这样同一线程的 bash 调用会累积目录变化。

为什么用线程局部变量？因为并行执行时，多个线程的 bash 调用不能共享同一个工作目录——否则一个线程 `cd` 了，另一个线程的下一行命令就跑到了错误的目录。

### agent.py — 子 Agent 调度

**我要解决什么问题**：有时候主 Agent 的任务可以拆成子任务——比如"你先去调查这个代码库的结构，然后回来告诉我"。子任务有独立的上下文窗口，不会污染主 Agent 的对话历史。

**实现**：接收一个 task 字符串。执行时创建新的 `Agent` 实例，传入和父 Agent 相同的 LLM、工具列表（但排除 `agent` 工具本身，防止无限递归）、相同的 max_context_tokens。子 Agent 最多 20 轮，返回的结果超过 5000 字符就截断。

**一个注意**：子 Agent 创建时需要把父 Agent 的引用传给它——我在 `Agent.__init__` 里做了：如果工具列表里有 `AgentTool`，把 `_parent_agent = self` 赋给它。

---

## 5. modes.py — 模式系统

工具写好了，但我觉得不能所有人都能用所有工具。

**我要解决什么问题**：不是每个用户都需要 bash 权限。读代码的人应该只有读的权限，改文档的人不应该能执行命令。我需要一个机制来按场景裁剪工具列表。

**怎么实现的**：一个字典 `MODE_TOOLS`，key 是模式名，value 是允许的工具名列表。

```
full:     所有 9 个工具
review:   read_file + grep + glob（只读）
coding:   read/write/edit/bash + 搜索（全权限）
document: read/write/edit + 搜索（无 bash）
infra:    read/grep/glob + gpu_status + vllm_status（只读诊断）
```

`get_tools_for_mode(mode)` 从字典里查模式名，拿到工具名列表后从注册表取实例。

**切换模式时发生了什么**：Agent 的 `set_mode()` 方法重新设置 `self.tools` 和 `self._tool_by_name` 映射，然后重新生成 system prompt——因为 prompt 里的工具列表是动态拼接的。这样 LLM 在 review 模式下根本不知道有 bash 这个工具存在，不是"不能用"，而是"不知道"，更安全。

---

## 6. prompt.py — 系统提示词

**实现**：`system_prompt(tools)` 函数接收当前可用的工具列表，动态生成 system prompt。内容包括：

- 角色定义：你是一个 AI 编程助手
- 环境信息：当前工作目录、操作系统、Python 版本
- 工具列表：每个工具的名字和描述
- 8 条行为规则：先读后写、小改动用 edit_file、改完后验证、保持简洁、顺序执行等

每次切换模式时 prompt 重新生成，工具列表跟着变。

---

## 7. agent.py — Agent 主循环

前面的模块都搭好了，现在把核心循环组装起来。

**核心思路**：一个 `for _ in range(self.max_rounds)` 循环，默认上限 50 轮。每轮做的事：

1. 拼完整上下文：system prompt + 历史消息
2. 调 LLM，传入 tools schema
3. 检查返回结果：
   - 没有 `tool_calls` → 说明 LLM 直接生成了文本回复，追加到历史，返回
   - 有 `tool_calls` → 执行工具，结果回填，继续循环
4. 50 轮用完还没结束 → 返回 `(reached maximum tool-call rounds)`

**为什么用 for 循环而不是 while True**：一道刹车。如果模型陷入一个调用工具的循环（读文件→发现不对→再读→还是不对→再读），没有上限它会一直烧 token。50 轮是经验值，正常任务通常在 5-10 轮内结束。

**并行执行**：当模型一次返回多个 `tool_calls` 时，它们之间没有依赖关系，可以用 `ThreadPoolExecutor`（最多 8 个线程）并发执行。单工具调用直接在当前线程执行。

**中断安全**：这是一个容易忽略的细节。如果用户在工具执行过程中按了 Ctrl+C，工具执行了一半就停了，`assistant` 消息里有 `tool_calls` 但没有对应的 `tool` 回复。下一次请求时 OpenAI 兼容 API 会拒绝——因为要求每条 tool 消息前面必须有对应的 assistant 消息（有 tool_calls 的那种）。我的 `_answer_pending_tool_calls` 方法会为所有未完成的 tool_call 回填 `[interrupted]` 消息，保证历史结构的完整性。

**工具参数校验**：在真正执行工具前，我用 `inspect.signature(tool.execute).bind(**tc.arguments)` 校验参数是否符合工具定义的签名。这样如果 LLM 传了错误的参数，错误信息会明确说"参数不对"而不是"执行出错"，方便 LLM 修正后重试。

**模式切换**：`set_mode()` 直接重新设置工具列表和 prompt，子 Agent 的父引用也重新绑定。

---

## 8. context.py — 上下文管理

Agent 循环跑起来了，但跑了几轮之后历史消息会越来越多。我需要一套机制来管理上下文窗口。

**我的思路**：Claude Code 用四层压缩，我简化成三层，从轻到重，惰性触发——不到阈值不做。

**第一层（50% 窗口）——截断工具输出**：工具调用返回的结果可能很大（比如 `cat` 了一个长文件）。超过 1500 字符的工具输出被截断，只保留头 3 行和尾 3 行，中间用 `... (N lines, snipped to save context) ...` 代替。这一层纯文本处理，不调 LLM。

**第二层（70% 窗口）——LLM 摘要**：把早期的对话整段交给 LLM，让它写一个简要摘要，只保留最近 8 轮消息完整不动。这里有一个很隐蔽的坑：如果我的分割点恰好落在一条 tool 回复上，这条 tool 消息前面应该有一条带 `tool_calls` 的 assistant 消息，但那 assistant 消息被切到前面了。下次请求时 API 会拒绝——因为 tool 消息找不到对应的 `tool_calls`。解决办法很简单，就一个 `while` 循环：分割前把边界往回退，直到边界不再是一条 tool 消息。

如果 LLM 不可用（比如摘要生成失败），我回退到 `_extract_key_info` 方法——从消息中提取文件路径和错误信息，拼成一段简短的摘要。不一定精确，但总比什么都不做要好。

**第三层（90% 窗口）——硬折叠**：窗口用到 90% 时紧急措施，只保留最近几条消息加一次全局摘要，其余全部丢弃。这是最后一道防线，很少会触发。

---

## 9. session.py — 会话持久化

现在 Agent 可以正常对话了，但会话只能在内存中存活。我需要把对话保存下来，下次可以恢复。

**实现**：对话历史序列化成 JSON，保存到 `~/.infracoder/sessions/` 目录下。文件名的格式是 `session_时间戳_随机hex.json`，也可以指定自定义 session ID。

**安全性**：session ID 可能被用来做路径穿越攻击（比如传入 `../../etc/passwd` 作为 session ID）。我在 `_session_path` 方法里用 `resolve()` 后检查父目录是否等于 `SESSIONS_DIR`，不相等就拒绝。同时用正则过滤，只保留 `[A-Za-z0-9._-]` 字符。

---

## 10. cli.py — 命令行界面

基础功能写完了，我需要一个和用户交互的界面。我决定先做 CLI，再做 Web。

**实现**：基于 `prompt_toolkit`（处理终端输入，支持多行编辑和历史记录）和 `rich`（终端样式美化）。

**交互模式**：
- 多行输入：Enter 提交，Esc+Enter 换行（方便粘贴代码块）
- 流式输出：LLM 生成的 token 逐个打印，用户能实时看到模型在思考
- 工具调用可视化：工具执行时用灰色小字打印工具名和参数

**内置命令**：`/model` 切换模型、`/mode` 切换模式、`/compact` 手动压缩、`/tokens` 看 token 用量和费用、`/diff` 看改了哪些文件、`/save` 和 `/sessions` 管理会话、`/reset` 清空对话。

**为什么有些命令以 `/` 开头**：避免与用户的普通输入混淆。如果用户输入了一个未知的 `/` 命令，我直接提示"试试 /help"而不是发给 LLM。

---

## 11. webui.py — Web 界面

CLI 对开发者和习惯终端的人好用，但部门里的大部分同事想要 Web 界面，打开浏览器就能用。

**实现**：基于 Gradio 6.x。每个浏览器标签页通过 `gr.State` 维护独立的 Agent 实例，互不干扰。侧边栏有一个实时状态面板，显示 vLLM 服务状态和 GPU 指标。

**流式对话**：Gradio 支持通过回调函数逐步更新聊天框。我把 Agent 的 `on_token` 回调接到 Gradio 的界面更新上，实现逐字显示。

**工具调用可视化**：工具执行时以灰色小字展示在消息中，格式是 `工具名(json 参数)`，让用户知道 Agent 正在做什么。

**状态刷新**：`_system_status` 函数用 `urllib` 请求 vLLM 的 `/v1/models` 端点检查服务状态，同时调用 `nvidia-smi` 获取 GPU 信息。每次刷新都是独立查询，不缓存状态。

---

## 12. tests/ — 测试

功能都搭起来了，最后补测试。我分了四个测试文件：

**test_core.py**：测试 Config、Context、Session 等核心模块。最关键的测试是验证压缩后不会产生孤立的 tool 消息——我构造了一个包含多条 tool 消息的对话，调用压缩后遍历消息列表，确保每条 tool 消息前面要么是一条 tool 消息，要么是一条带 tool_calls 的 assistant 消息。

**test_tools.py**：测试所有 9 个工具。包括：
- bash：危险命令拦截、超时、输出截断、多线程 cd 隔离
- read_file：行号范围、偏移量、Unicode 编码
- write_file：自动创建目录
- edit_file：0 次匹配、多次匹配、二进制文件拒绝
- grep：跳过内部目录、祖先目录不跳过

**test_session.py**：测试会话 ID 碰撞、路径穿越防护、绝对路径剥离、长度上限、Unicode 编码回环。

**test_litellm.py**：Mock LiteLLM 的 API 返回，测试 LiteLLM 后端的参数传递、模型字符串路由、token 统计。

---

## 总结：搭建顺序

如果你打算从头复现这个项目，推荐的搭建顺序是：

1. **config.py** — 配置（最独立）
2. **llm.py** — LLM 通信
3. **tools/base.py** — 工具基类
4. **tools/read.py → write.py → edit.py → grep.py → glob_tool.py → bash.py → agent.py → search_knowledge.py → workflow.py** — 具体工具，从简单到复杂
5. **modes.py** — 模式系统
6. **prompt.py** — 提示词（含个性化输出风格）
7. **agent.py** — 核心循环（组装前面的模块）
8. **context.py** — 上下文管理
9. **session.py** — 会话持久化
10. **cli.py** — 命令行界面（含 kb、profile 命令）
11. **webui.py** — Web 界面
12. **workflows/** — 工作流模板系统
13. **knowledge/** — RAG 知识库
14. **user_config.py** — 用户个性化配置
15. **tests/** — 测试

每个文件的逻辑都独立，搭好前一个就可以调试下一个。我从 config 开始，写完 llm.py 之后手动构造消息测试通信正常，然后逐个加工具、加循环、加压缩、加界面，每一步都可以独立验证。
