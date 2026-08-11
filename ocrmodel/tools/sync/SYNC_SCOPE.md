# 服务器同步范围

`sync_to_server.ps1` 只同步 Git 可见且位于以下目录的文件：

- `src/`
- `tools/`
- `config/`
- `references/`

`config/paths.env` 被 `.gitignore` 排除，不会覆盖服务器的机器本地配置。脚本也不会同步或删除以下内容：

- `archive/`、`results/` 与旧迁移材料；
- `.venv/`、缓存和编辑器配置；
- 数据、模型、checkpoint、日志和 run 目录；
- 私钥、密码或 SSH 配置；
- 远端已有但本地已删除的陈旧文件。

远端文件删除或重命名必须按明确路径单独处理，不执行目录级清空。

不连接服务器的范围检查：

```powershell
.\tools\sync\sync_to_server.ps1 `
  -RemoteHost placeholder `
  -RemoteRoot /tmp/ocrmodel `
  -DryRun
```
