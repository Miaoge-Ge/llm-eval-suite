# API60 与思考强度

网页和 `llm-eval api-test` 默认运行 `api-experience-v2`，六类场景各 10 题，交错执行，共 60 次独立流式请求。每题只有一次尝试，没有自动降档、重试或跨时段续跑。

| 场景 | 观测目的 | 输出检查 |
| --- | --- | --- |
| 短回答 | 首段响应与短请求完成时间 | 不判答案正误 |
| 长回答 | 持续流式输出、停顿与吞吐 | 不判答案正误 |
| 长上下文 | 约 4K 参考 token 输入下的响应 | 不判答案正误 |
| JSON 输出 | `response_format: {"type":"json_object"}` 的接口兼容性 | 严格 JSON 对象、字段和值 |
| 多轮上下文 | system/user/assistant 历史消息与后续修正 | 返回最新项目编号 |
| 中英文混合 | 中文、英文、重音字符、标点与 emoji 输出 | 精确回显（忽略首尾空白） |

输出检查通过率单独统计，不改变流式请求成功率，也不能单独归因于接口或模型。失败流、截断流没有已完成的输出检查。接口拒绝 JSON 模式会如实记录为 API 错误，不自动改成普通请求。每组只有 10 次，P95 只是当前样本统计，不能作为长期 SLA。

逐段时间戳、首段有效内容、首段正文、截断、断流、限流、用量缺失和独立调度等待继续保留。只有思考内容或缺少可靠 token/时间信息时，正文 TTFT/TPS 显示缺失。旧 API30 原始记录可离线重算，报告题数由保存的记录决定，不会标成 60 题。

## OpenAI 兼容中转的思考强度

| 模型 | 可选档位 |
| --- | --- |
| deepseek-v4.1-flash | none、low、high、max |
| deepseek-v4-pro | none、low、high、max |
| qwen3.8-flash | none、low、medium、xhigh |
| glm-5.3-flash | low、high、max |
| mimo-v2.6-flash | none、low、high |
| mimo-v2.6-pro | none、low、high |
| MiniMax-M3.1-Flash | low、medium、high、xhigh、max |

这些预设来自用户指定的清单，不是对官方 API 或中转能力的认证。自定义模型仍可输入，并选择通用档位。命名空间前缀（如 `gateway/model`）和大小写不影响预设匹配；发送给中转的模型名称保持原样。

显式档位作为 `reasoning_effort` 原样发送，`none` 表示请求关闭思考；不再叠加厂商专属 thinking 开关。网页的“默认强度（启用思考）”保留原请求方式 `thinking.type=enabled`，不发送 effort。模型切换后若旧档位不可用，网页恢复默认并提示。服务端也会拒绝已知模型不支持的预设组合。

中转可能拒绝或忽略参数；报告记录的是请求值，不能证明服务端实际执行。HTTP 400/422 不会触发降档重试。任务列表、详情、配置快照、原始请求和报告均保留所选强度。

CLI 示例：

```yaml
thinking_enabled: true
reasoning_effort: high
```

请求关闭思考时设 `reasoning_effort: none`。未指定 effort 的旧配置保持兼容。Quick40 题包、任务和 `quick-test` 命令已经移除；Benchmark 直接选择标准题库。
