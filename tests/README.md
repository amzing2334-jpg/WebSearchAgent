# WebSearchAgent 离线程序逻辑测试

这些测试用**预先固定的输入**验证**真实的 Python 程序逻辑**，不是联网问答演示，也不是模型效果评测。

不加载 `.env`，不需要 API Key，不访问 Serper、Jina、百炼或 Hugging Face。新测试还会阻断 Requests、HTTPX 和底层 socket/DNS 网络入口；即使业务代码吞掉了阻断异常，测试退出时仍会失败。网络守卫只记录入口名，不记录 URL、请求头或密钥。

## 先运行这一条：检查完整 Agent 流程

在 VS Code 的 PowerShell 终端执行（从任意目录均可）：

```powershell
python -B -m unittest discover -s tests -p 'test_agent_flow_offline.py' -v
```

请使用已安装 `requirements.txt` 依赖的 Python 解释器。`-B` 表示不生成新的 `.pyc` 缓存；没有安装新的测试框架，使用的是 Python 自带的 `unittest`。

**原有 F5 的 `WebSearchAgent` 配置仍会启动真实联网的 `run_local.py`。它没有被修改，不是这里的离线测试入口。**

## 哪些是真的，哪些是固定返回？

| 保留真实执行 | 用测试数据替代 |
| --- | --- |
| `_run()` 主循环、条件判断、JSON 解析 | 主模型 `call_server()` 的返回字符串 |
| `decompose_question()` 和 `update_scratchpad()` 的提示拼接、结果处理 | `_decomposer_call()` 返回的计划、校验结果和摘要 |
| 工具注册表查找、实例化、框架 `_call_tool()` 派发 | Search、Visit 实例的 `.call()` 返回值 |
| 消息追加、system 更新、Reminder 和历史裁剪 | `count_tokens()` 返回固定小值，避免下载分词器 |

测试用 `object.__new__` 跳过 Agent 中的模型初始化，再调用真实 `_init_tool()` 初始化 Search 和 Visit。因此覆盖工具注册查找和派发，但不覆盖 Agent 构造器的全部行为。

`test_visit_error_paths_offline.py` 是另一组测试：它执行真实 `Visit.call_server()` 和 `readpage()`，只替换 OpenAI 客户端回复和网页读取。

## 六次工具调用的固定流程

固定问题、计划和网页均是测试夹具；最终“测试答案”不是真实知识答案。

| 主模型轮次 | 固定返回 | 本轮开始时的消息数 | 处理结果 |
| --- | --- | ---: | --- |
| 1 | Search 查询 1 | 2 | 写回后 4 条 |
| 2 | Visit 网页 2 | 4 | 写回后 6 条 |
| 3 | Search 查询 3 | 6 | 写回后 8 条；首次摘要；不裁剪 |
| 4 | Visit 网页 4 | 8 | 写回后 10 条 |
| 5 | Search 查询 5 | 10 | 写回后 12 条 |
| 6 | Visit 网页 6 | 12 | 写回后 14 条；第二次摘要；裁剪为 8 条 |
| 7 | 带 `<verify>` 的最终答案 | 8 | 追加最终 assistant，结果中共有 9 条 |

每次请求前都对 `messages` 做 `deepcopy`，保存**当时的快照**。不能直接保存原列表引用：后面的原地更新会把“历史记录”一起改掉。

测试不仅检查数量，还会核对：

- Search/Visit 选择、顺序、参数和调用次数；是否写回正确工具的唯一结果标记。
- 主模型每次看到的消息数必须为 `[2, 4, 6, 8, 10, 12, 8]`。
- 摘要只能在工具调用 3、6 后执行，新结果批次分别是 1–3、4–6；第二次摘要输入必须带上旧摘要。
- 裁剪后必须保留原 system 规则、原始 user，以及调用 4、5、6 的消息对，不能仅满足“长度为 8”。
- 最后返回 `prediction == '测试答案'`，不拿模型输出的语义质量代替程序判断。

另外包含：无计划、非法 JSON、未知工具、模型伪造工具结果、缺失 verify、摘要失败，以及恰好 8 条/超过边界到 9 条时的检查。

## 如何查看和修改测试

打开 `test_agent_flow_offline.py`，按这个顺序阅读：

1. `six_tool_calls()`：预先准备六次工具名与参数。
2. `tool_output()`：把它们转换为项目实际接收的 `<tool_call>` 字符串。
3. `ScriptedRun.main_response()`：每调用一次，返回下一条，并记录输入快照。
4. `ScriptedRun.run()`：只替换外部边界，然后实际执行 `self.agent._run(...)`。
5. `AgentFlowOfflineTests`：用 `assertEqual` 等比较实际与预期，不一致就失败。

例如，把程序裁剪错写为“只保留前 8 条”，长度可能仍然正确，但 `test_history_is_not_trimmed_at_8_then_keeps_exact_calls_4_5_6` 会因为消息内容不符而失败。**不要为了让测试变绿，把合理预期改成错误程序当前的表现。**

要在 VS Code 里逐步看测试，可在 `ScriptedRun.run()` 中实际调用 `_run()` 的那一行打断点，使用调试“当前 Python 文件”的方式启动这个测试文件；不要误选原来的联网启动配置。也可以先仅用上述终端命令，阅读失败时显示的源文件与行号。

## 单独运行错误分支测试

```powershell
python -B -m unittest discover -s tests -p 'test_visit_error_paths_offline.py' -v
```

当前业务代码中已复现的两处问题，特意保留为真实的失败测试，**没有跳过，没有标记 expectedFailure，也没有改业务逻辑**：

1. 模型连续空回复：`call_server()` 最终隐式返回 `None`，`readpage()` 在 `len(raw)` 处崩溃。预期是返回可处理的失败说明。
2. JSON 解析最后一次重试：已收到第四份合法 JSON，但没有解析就进入失败分支。预期是使用这份已请求到的合法结果。

同一文件还有两个成功对照：首次合法 JSON，以及第三份回复恢复为合法 JSON。

## 运行项目全部单测

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

当前预期为 34 项：32 项通过、1 项 FAIL、1 项 ERROR。全部测试命令退出码为 1，原因是上面两处真实业务缺陷，不是 API、额度或网络问题。

- `ok`：这个测试验证的预期成立。
- `FAIL`：程序运行了，但实际结果不符合断言。
- `ERROR`：执行过程中发生了未处理异常。

以后修复业务逻辑后，重新运行同一批测试，确认这两项也通过，并且其它项没有被破坏。这就是回归测试。

## 测试边界

本套流程测试不证明模型推理、搜索质量、网页真实性、摘要语义或实际 token 计数正确；也没有覆盖全部异常、并发和重启恢复场景。正常用例通过不代表项目无 bug。真实联网端到端测试仍然需要单独进行，并与这些模拟输入测试明确区分。
