---
name: Bug report
about: Something crashes, returns wrong results, or loses accuracy
title: ""
labels: bug
assignees: ""
---

**What happened?**

A clear description of the bug.

**Minimal reproducer**

```julia
using TwoDG
# the smallest script that shows the problem
```

**Expected behavior**

What you expected instead. For accuracy bugs, the observed vs expected
convergence rates (or errors across a refinement sequence) are the most
useful evidence.

**Environment**

- TwoDG.jl version / commit:
- Julia version (`versioninfo()`):
- OS:
- GPU involved? (`CUDA.versioninfo()` if so):
