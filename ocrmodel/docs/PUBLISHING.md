# 共享远程发布

当前开发目录嵌在一个更大的父 Git 仓库中。父仓库历史包含不属于 OCR 共享范围的申报材料、文献和个人记录。根 `.gitignore` 只能限制后续工作树和提交，不能从已有历史中移除这些文件。

因此，共享远程不得直接接收父仓库 `master`。应发布只包含 `ocrmodel/` 历史的 subtree 分支，使本目录内容成为远程仓库根目录。

## 首次发布

先由维护者审查并提交本次变更。创建共享分支：

```powershell
git subtree split --prefix=ocrmodel -b shared/ocrmodel
```

发布前检查根目录和历史文件名：

```powershell
git ls-tree --name-only shared/ocrmodel
git log --name-only --format= shared/ocrmodel | Sort-Object -Unique
```

输出只应出现本仓库的 `README.md`、`src/`、`tools/`、`docs/`、`config/`、`tests/` 和 `references/` 等共享内容，不应出现父项目路径。

向空的共享远程发布；替换远程名和 URL：

```powershell
git remote add ocr-shared '<remote-url>'
git push -u ocr-shared shared/ocrmodel:main
```

不要使用 `--force` 覆盖已有协作者历史。若远程不是空仓库，应先核对其分支和所有权。

## 后续发布

父仓库完成新的 OCR 提交后，再次生成 subtree 分支并正常推送。若本地 `shared/ocrmodel` 已存在，可先选择新的临时分支名，不需要改写父仓库历史：

```powershell
git subtree split --prefix=ocrmodel -b shared/ocrmodel-update
git push ocr-shared shared/ocrmodel-update:main
```

共享仓库的协作者 clone 后看到的仓库根目录就是当前 `ocrmodel/` 内容，文档中的相对命令可直接使用。

## 发布前检查

至少确认：

- `git status` 中没有密钥、`config/paths.env`、数据、权重、日志或结果；
- `rg` 检查不到个人服务器地址、用户名和私有绝对路径；
- `archive/`、`results/`、旧迁移文档与父项目文件不在 subtree 分支；
- 本地语法检查和 CPU 测试通过；
- 当前失败诊断、未完成项和结果边界与 `docs/PROJECT_STATUS.md` 一致。
