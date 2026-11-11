# Issue tracker：GitHub

这个 repo 的 issues 和 specs 存放在
`Lin-Ruipeng/IMU-predict` GitHub Issues 中。所有操作都使用 `gh` CLI。

## 约定

- **Verify the target**: session 第一次写入前运行
  `gh repo view --json nameWithOwner`，结果必须是
  `Lin-Ruipeng/IMU-predict`。
- **Create an issue**: `gh issue create --title "..." --body "..."`。多行 body
  使用 heredoc。
- **Read an issue**:
  `gh issue view <number> --json number,title,body,labels,comments`。
- **List issues**: 使用 `gh issue list --state open`，按需添加 `--label` 和
  `--state` filters。
- **Make an issue a sub-issue**: 使用
  `gh issue create --parent <parent> ...`，或
  `gh issue edit <parent> --add-sub-issue <child>`。无法使用 sub-issues 时，在
  child body 顶部写 `Part of #<parent>`。
- **Blocking**: 优先使用 GitHub native issue dependencies：
  `gh issue create --blocked-by <n>,<n> ...` 或
  `gh issue edit <child> --add-blocked-by <blocker>`。不可用时，在 child body
  顶部写 `Blocked by: #<n>, #<n>`。
- **Comment**: `gh issue comment <number> --body "..."`。
- **Apply / remove labels**:
  `gh issue edit <number> --add-label "..."` /
  `--remove-label "..."`。
- **Close**: 先发布说明，再运行 `gh issue close <number>`。

只把 issues 提交到用户自己的 repo。绝不向 skills 的来源 repo 提交 issue。

## Pull requests 作为 triage surface

**PRs as a request surface: no.**

GitHub 在 issues 和 PRs 之间共享 number space；裸 `#42` 可能是两者之一。

## 当 skill 说 “publish to the issue tracker” 时

创建一个 GitHub issue。

## 当 skill 说 “fetch the relevant ticket” 时

按上方 **Read an issue** 的方式读取。

## Wayfinding 操作

供 `wayfinder` 使用。map 是单个 issue，以 child issues 作为 tickets。

- **Map**: 使用带 `wayfinder:map` label 的 issue 保存 Notes、
  Decisions-so-far 和 Fog。
- **Child ticket**: 作为 GitHub sub-issue 链接到 map；不可用时使用 task list
  与 `Part of #<map>`。
- **Blocking**: 使用上方的 native dependency 约定。
- **Claim**: `gh issue edit <n> --add-assignee @me`。
- **Resolve**: 发布答案、关闭 child，再向 map 追加 context pointer。
