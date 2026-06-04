# 🤖 Agent Playbook: Technical Handoff & Architecture Decisions

This file is written by an AI Coding Agent to guide subsequent AI agents taking over maintenance, expansion, or troubleshooting of the **vLLM Internals** repository. It highlights critical design decisions, hardware/API gotchas, and layout structures to save context and prevent repeating resolved issues.

---

## 1. Repository Overview & Scope
*   **Target User**: AI Serving Engineers, Deep Learning Engineers, and Research Engineers.
*   **Mission**: A structured textbook/lecture series analyzing the internals of the `vLLM` library (scheduling, memory, block management, multi-GPU orchestration), accompanied by a simplified Python simulator of a continuous batching serving engine.
*   **Deployment**: Hosted on GitHub Pages at `https://tuandung222.github.io/vllm-architecture-lectures/`.
*   **Git Author Identity Constraint**: Local git commits MUST be authored by `tuandung222` to maintain proper contributor attribution.
    *   Command: `git config user.name "tuandung222" && git config user.email "75377334+tuandung222@users.noreply.github.com"`

---

## 2. Directory Layout & Critical Decoupling
To avoid folder pollution and framework build conflicts, the repository is strictly divided:

```bash
vllm-architecture-lectures/
├── docs/                      # Docusaurus Markdown Lectures
│   ├── roadmap.md             # Lesson 0: Roadmap (Position: 0)
│   ├── lesson_0_*.md          # Lesson 0/0.1: OS & GPU Hardware Background (Position: 1, 2)
│   ├── lesson_1_*.md          # Lesson 1/1.1: Autoregressive & Arithmetic Intensity (Position: 3, 3.5)
│   ├── lesson_2_*.md          # Lesson 2/2.1: PagedAttention & Compute Backends (Position: 4, 4.5)
│   └── ...                    # Lessons 3-8 (Continuous Batching, Async, Code Deep Dives)
├── toy_engine/                # Python Simulator Code (Replaced old "src/")
│   ├── allocator.py           # Logical-to-Physical page/block allocator
│   ├── scheduler.py           # Continuous batching scheduler
│   ├── model.py               # Mock model simulating Prefill vs Decode delays
│   ├── app.py                 # FastAPI engine async runner & SSE streaming server
│   └── client.py              # Test script sending concurrent prompts & testing aborts
├── src/                       # Docusaurus React Source (DO NOT put Python code here)
├── static/                    # Docusaurus Static Assets
├── .github/workflows/         # CI/CD deployment pipelines
└── docusaurus.config.ts       # Global site configurations
```

### ⚠️ Critical Gotcha 1: The Rename of `src/` to `toy_engine/`
*   **Problem**: Docusaurus is a React framework that expects its React components and website pages to live in `src/` by default. Initially, the Python simulator code for Lesson 8 lived in a folder named `src/`. This created immediate conflicts during Docusaurus builds.
*   **Solution**: The Python simulator folder was renamed to `toy_engine/`. 
*   **Action for incoming agents**: All Python imports in `toy_engine/*.py` must import from `toy_engine.xyz` instead of `src.xyz`. Do not put any Python files back into `src/`.

---

## 3. GitHub Pages & API Quirks (Important)
*   **Problem**: When configuring GitHub Pages via the GitHub REST API (`gh api`), sending a `PATCH` request to `repos/{owner}/{repo}/pages` to change the source branch from `main` to `gh-pages` returns a **404 Not Found** error (even with correct admin tokens).
*   **Root Cause**: The GitHub REST API has undocumented quirks where changing build source branches is blocked or misrouted under `PATCH` for legacy pages setups.
*   **Workaround**: Use a **`PUT`** request instead. Send a `PUT` request with the JSON payload:
    ```json
    {
      "source": {
        "branch": "gh-pages",
        "path": "/"
      }
    }
    ```
    This returns a successful **HTTP 204 No Content** and successfully updates the source branch.

---

## 4. KaTeX CSS Subresource Integrity (SRI)
*   **Problem**: Math formulas inside Markdown were showing "double rendering" (showing both raw LaTeX letters/MathML tags stacked, e.g. `N` and then `N=1` right below it).
*   **Root Cause**: The KaTeX CSS stylesheet link (`katex.min.css`) failed browser Subresource Integrity (SRI) checks due to a mismatched hash, preventing the CSS from loading. Without CSS, the browser does not hide the screen-reader MathML tags.
*   **Solution**: Ensure `docusaurus.config.ts` uses the exact correct SRI hash matching the version. For KaTeX `0.16.8`, the correct configuration is:
    ```typescript
    stylesheets: [
      {
        href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css',
        type: 'text/css',
        integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn',
        crossorigin: 'anonymous',
      },
    ]
    ```

---

## 5. Docusaurus Configuration Decisions
*   **Floating-Point Sidebar Ordering**: Docusaurus supports floating-point values for `sidebar_position`. Supplementary files are ordered using decimals (e.g. `lesson_1_1_*.md` has `sidebar_position: 3.5` to sit cleanly between Lesson 1 (3) and Lesson 2 (4)) without shifting other files.
*   **Blog Disabled**: `blog: false` is set in Docusaurus classic presets to keep the interface purely documentation-focused.
*   **Mermaid Diagrams**: `@docusaurus/theme-mermaid` is installed. Ensure `markdown: { mermaid: true }` and `themes: ['@docusaurus/theme-mermaid']` are present in configuration.

---

## 6. Active Tasks & Backlog
If the user asks to continue developing the project, here is the current backlog:
*   [ ] Expand **Lesson 5 & 6** codebase deep dives as vLLM v1 matures (specifically focusing on `vllm/v1/core/sched/request_queue.py` and `vllm/v1/engine/core.py`).
*   [ ] Enhance the `toy_engine` simulator to support *Chunked Prefill* (currently simulated inside Lesson 3 markdown but not implemented in the Python code).
*   [ ] Implement a simple *Swapping* simulation in `toy_engine/scheduler.py` via PCIe bandwidth delay emulation.
