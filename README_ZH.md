# CrabRAG

CrabRAG 是一个本地 RAG 应用，包含 Python FastAPI 后端、Bun 网关、已打包 Web 界面，以及 CLI 证据检索入口。

默认地址：

- Web 界面：`http://127.0.0.1:3003/`
- Python API：`http://127.0.0.1:8001/`

## 环境要求

- Python 3.10 或更高版本
- Bun 可在 `PATH` 中访问
- 首次安装依赖时需要联网

Node.js、npm 和 pnpm 会被安装脚本检测并用于诊断提示，但当前打包 Web 网关使用 Bun。

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

如果缺少 Bun，先安装：

```bash
curl -fsSL https://bun.sh/install | bash
```

安装 Bun 后需要重启终端，确保 `bun` 已加入 `PATH`。

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

重新安装 JavaScript 依赖：

```bash
bun install
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

## 常见问题

- `Python 3.10+ was not found`：安装 Python 3.10 或更高版本，然后重新运行安装脚本。
- `Bun was not found`：安装 Bun，并重启终端。
- `Local ONNX runtime unavailable`：远程/API 模式仍可运行；只有需要本地向量或本地重排模型时才需要修复 ONNX runtime。
- `API port 8001 is already in use`：停止占用该端口的进程，或改用其他 API 端口。
- `Web port 3003 is already in use`：停止占用该端口的进程，或改用其他 Web 端口。
- `config/.env is missing`：重新运行安装脚本，或手动复制 `config/.env.example` 为 `config/.env`。
- 页面能打开但 API 请求失败：确认 Python API 正在运行，并访问 `http://127.0.0.1:8001/api/health` 检查健康状态。
