# CrabRAG

CrabRAG 是一个本地 RAG 应用，包含 Python FastAPI 后端、Bun 网关、已打包 Web 界面，以及 CLI 证据检索入口。

默认地址：

- Web 界面：`http://127.0.0.1:3003/`
- Python API：`http://127.0.0.1:8001/`

## 环境要求

- Python 3.10 或更高版本
- 首次安装依赖时需要联网

安装脚本会使用 Bun 启动打包 Web 网关。如果系统 `PATH` 中没有 Bun，安装脚本会自动下载项目本地 Bun 到 `runtime/bun/`。Node.js、npm 和 pnpm 仅用于诊断提示。

## Windows 安装

在项目目录中打开 PowerShell：

```powershell
.\install.ps1
.\run.ps1
```

也可以使用以下命令启动：

```powershell
.\start.bat
```

如果 PowerShell 阻止脚本执行，使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

## Linux 安装

在项目目录中执行：

```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
```

如果缺少 Python 或 venv 支持，先安装：

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y python3 python3-venv python3-pip

# CentOS / Rocky / AlmaLinux
sudo dnf install -y python3 python3-pip
```

如果系统 `PATH` 中没有 Bun，安装脚本会自动下载项目本地 Bun 到 `runtime/bun/`。

## 配置

安装脚本只会在 `config/.env` 不存在时，将 `config/.env.example` 复制为 `config/.env`。如果已有配置，安装脚本不会覆盖。

常用配置：

- `CRABRAG_DOCS_DIR`：可选知识库目录，多个目录用 `;` 分隔。
- `RAG_BASE_URL`：手动启动服务时，Bun 网关访问的 Python API 地址。
- `PORT`：手动启动 `server/gateway.js` 时使用的 Web 网关端口。

模型 API Key 和聊天模型设置通常在页面的“设置”中配置，不需要写入 `config/.env`。

## 本地模型

本地模型是可选功能。如果在设置页启用本地模型，需要手动下载模型并放入：

```text
runtime/models/
```

设置页会检测缺失的模型文件，并根据当前语言显示对应的 ModelScope 或 Hugging Face 下载链接。
ONNX runtime 只在启用本地向量模型或本地重排模型时需要。远程/API 模式下，即使当前机器无法导入 ONNX runtime，也不应影响项目启动。

## 安装检测

安装完成后，可以运行：

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

Linux：

```bash
./.venv/bin/python scripts/check_env.py
```

检测内容包括关键 Python 包、`config/.env`、必要目录、已打包 Web 界面、网关入口和 Bun 是否可用。
如果看到 ONNX runtime 不可用的可选警告，表示本地 ONNX 模型能力不可用，但不会阻断远程/API 模式运行。

## 重新安装依赖

重新安装 Python 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## 开发与测试

正常安装和运行环境仍使用 `requirements.txt`。开发时通过 `requirements-dev.txt` 同时安装运行依赖和已锁定版本的测试依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```

重新安装 JavaScript 依赖：

```bash
runtime/bun/bun install
```

Windows 下使用 `runtime\bun\bun.exe install`。

源码仓库会保留 `tests/` 回归测试。Windows 用户发布包不会包含测试、React/TypeScript 源码、网关 TypeScript 源码、缓存、本地配置、用户数据、API Key 或本地模型。

从源码重建 Web 与网关，或执行完整检查：

```bash
bun run build
bun run check
```

## 诊断、备份与恢复

执行跨平台安装诊断（退出码 `0` 表示健康、`1` 表示存在告警、`2` 表示存在错误）：

```powershell
.\.venv\Scripts\python.exe scripts\crabrag_admin.py doctor --json
```

创建和恢复带校验的备份：

```powershell
.\.venv\Scripts\python.exe scripts\crabrag_admin.py backup --output .\backup.zip
.\.venv\Scripts\python.exe scripts\crabrag_admin.py restore --archive .\backup.zip --yes
```

备份包含本地配置、Chroma、索引 generation 和应用状态。外部知识库文件不会进入备份，只记录规范化后的目录路径。恢复前必须停止 CrabRAG；恢复会先检查格式、软件兼容性、路径边界和每个文件的 SHA-256，全部通过后才替换本地状态。
备份可能包含明文 API 凭据，因此命令会在 manifest 和输出中明确告警，并在操作系统支持时设置为仅所有者可访问。请像保管 API Key 一样安全保管备份 ZIP。

生成 Windows x64 发布包及 SHA-256 文件：

```powershell
.\scripts\build_release.ps1 -Version 1.1.0 -OutputDir .\release
```

如果需要从干净虚拟环境重新安装，可以删除 `.venv` 后重新执行安装脚本：

```powershell
Remove-Item -LiteralPath .\.venv -Recurse -Force
.\install.ps1
```

```bash
rm -rf .venv
./install.sh
```

不要删除 `data/`、`docs/` 或 `config/.env`，除非你明确想清理本地运行数据、知识库文件或配置。

## 生产级索引治理

CrabRAG 使用双代索引原子发布。增量构建在独立 generation 中完成文本向量、分类与图谱，全部成功后才切换活动指针，并保留上一成功代用于回滚。

每个知识库根目录可维护 `.crabrag-manifest.json`：

```json
{
  "schema_version": 1,
  "knowledge_base_id": "kb-example",
  "documents": [
    {
      "document_id": "policy-a",
      "path": "policy-v2.pdf",
      "version": "2",
      "status": "published",
      "effective_at": "2026-08-01T00:00:00Z",
      "updated_at": "2026-07-11T00:00:00Z",
      "acl": {"visibility": "restricted", "roles": ["sales"], "revision": "7"}
    }
  ]
}
```

状态可选 `draft`、`published`、`retired`。未登记文件会自动按 public/published/version 1 登记，并产生高风险治理告警。所有时间必须使用带时区的 ISO-8601 格式。

- `GET /api/index/status`：活动代、上一代、向量复用、缓存、调度和告警状态。
- `POST /api/index/rollback`：管理员回滚上一兼容代。
- `CRABRAG_INTERNAL_TOKEN`：Bun 网关与 RAG API 的内部信任令牌；标准启动脚本会在未配置时自动生成。

本地身份适配器读取 `CRABRAG_SUBJECT`、`CRABRAG_ROLES`、`CRABRAG_GROUPS`、`CRABRAG_PERMISSION_REVISION` 和 `CRABRAG_LOCAL_ADMIN`。浏览器自行提交的同名权限头不会被转发。

## 常见问题

- `Python 3.10+ was not found`：安装 Python 3.10 或更高版本，然后重新运行安装脚本。
- `Failed to install project-local Bun`：检查是否能访问 GitHub Releases，或从 <https://bun.sh/docs/installation> 手动安装 Bun。
- `Local ONNX runtime unavailable`：远程/API 模式仍可运行；只有需要本地向量或本地重排模型时才需要修复 ONNX runtime。
- `API port 8001 is already in use`：停止占用该端口的进程，或改用其他 API 端口。
- `Web port 3003 is already in use`：停止占用该端口的进程，或改用其他 Web 端口。
- `config/.env is missing`：重新运行安装脚本，或手动复制 `config/.env.example` 为 `config/.env`。
- 页面能打开但 API 请求失败：确认 Python API 正在运行，并访问 `http://127.0.0.1:8001/api/health` 检查健康状态。
