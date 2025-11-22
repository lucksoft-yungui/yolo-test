# yolo-test

## 环境设置 (Environment Setup)

本项目使用 [uv](https://github.com/astral-sh/uv) 进行包管理。

### 1. 安装 uv
如果你的机器上还没有安装 uv，请运行：

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 恢复环境
克隆代码或 `git pull` 后，在项目根目录运行以下命令即可一键安装 Python 版本及所有依赖：

```bash
uv sync
```

该命令会自动：
- 下载并安装所需的 Python 版本（由 `.python-version` 指定）。
- 创建 `.venv` 虚拟环境。
- 安装所有锁定在 `uv.lock` 中的依赖。

### 3. 运行代码
确保使用虚拟环境中的 Python：

```bash
uv run python main.py
# 或者激活环境后运行
source .venv/bin/activate
python main.py
```
