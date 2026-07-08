---
name: crabrag-rag
description: 查询本机 CrabRAG RAG 知识库，返回证据片段 JSON；当用户问题需要本地知识库资料、法规、制度、流程或业务规则依据时使用。
metadata:
  openclaw:
    requires:
      env:
        - CRABRAG_HOME
---

# CrabRAG 证据检索

当用户问题可能需要本机 CrabRAG 知识库中的资料、制度、流程、业务规则或证据片段时，使用此 Skill。UniClaw 单文件入口为项目根目录的 `crabrag.skill`；本文件保留给支持目录式 Skill 的本地调用端使用。

## 前置配置

`CRABRAG_HOME` 必须指向 CrabRAG 项目根目录，例如：

```bat
set CRABRAG_HOME=F:\trae\CrabRAG
```

## 查询证据

调用本地 CLI，并原样传入用户问题：

```bat
"%CRABRAG_HOME%\crab-rag.bat" --question "用户问题" --top-k 6 --pretty
```

可选参数：

- `--mode auto|vector|graph|hybrid`：选择检索方式。默认使用 `auto`。
- `--include-trace`：调试时返回检索诊断信息。
- `--no-rerank`：跳过 Rerank，适合只需要最快原始检索结果的场景。

## 回答规则

只根据 CLI 返回的 JSON 证据回答。把 `evidence[].content` 作为原文依据；有来源信息时，引用 `source_file` 和 `section_title`。

如果 `ok` 为 `false`，说明本地 RAG 查询失败，并带上 `error` 内容。

如果 `evidence` 为空，或 `warnings` 包含 `out_of_scope` / `no_evidence`，说明本地知识库没有提供足够依据。不要编造事实。

不要调用 CrabRAG 的 `/api/chat` 接口。本 Skill 只负责查资料和返回证据，最终回答必须由当前 Uniclaw/OpenClaw 模型根据证据生成。

## 使用要求

Uniclaw 回答要更直接，但不能牺牲证据约束：

- 先给结论，再列依据。
- 依据不足时直接说明“知识库未检索到足够证据”。
- 不要把常识、猜测或模型自身知识混入知识库结论。
- 来源引用保持简洁，例如：`来源：rules.md / 资费材料`。
