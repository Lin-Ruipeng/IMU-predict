# 领域文档

Engineering skills 探索 codebase 时，应如何消费这个 repo 的 domain
documentation。

## 探索前先读取这些

- repo 根目录的 **`GLOSSARY.md`**，或
- repo 根目录的 **`GLOSSARY-MAP.md`**（如果存在）：它指向每个 context 的
  一个 `GLOSSARY.md`。读取与当前话题相关的每个文件。
- **`docs/adr/`**：读取与你即将处理区域相关的 ADRs。在 multi-context repos
  中，也检查 `src/<context>/docs/adr/` 中的 context-scoped decisions。

如果这些文件不存在，**静默继续**。不要标记缺失；不要提前建议创建。
`domain-modeling` skill（经由 `grill-with-docs` 调用）会在 terms 或 decisions
实际被解决时懒创建它们。

## 文件结构

这个仓库使用 single-context 布局：

```text
/
├── GLOSSARY.md
├── docs/adr/
└── src/
```

`GLOSSARY.md` 和 `docs/adr/` 不需要预先存在。

## 使用 glossary 的词汇

当输出命名某个 domain concept 时（issue title、refactor proposal、hypothesis、
test name），使用 `GLOSSARY.md` 中定义的 term。不要漂移到 glossary 明确避免
的 synonyms。

如果需要的概念还不在 glossary 中，要么正在发明项目没有使用的语言，要么
确实存在缺口；后一种情况交给 `domain-modeling` 处理。

## 标记 ADR 冲突

如果输出与现有 ADR 矛盾，明确指出，而不是静默覆盖。
