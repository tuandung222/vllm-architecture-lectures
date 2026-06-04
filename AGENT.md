# 🤖 Agent Playbook: Architecture, Rigor, & Pedagogical Blueprint

This playbook is written by a senior AI Serving & Deep Learning Expert Agent for subsequent coding agents. It defines the architectural decisions, gotchas, and **pedagogical guidelines** for maintaining the **vLLM Internals** course repository.

---

## 1. Pedagogical Philosophy: Theory-to-Code Alignment
As an educator and systems engineer, our goal is to prevent students from "hand-waving" or treating LLM serving as a black box. Every theoretical concept (Roofline Model, memory layout, scheduling) must be mapped directly to:
1.  **The Mathematical & Hardware reality** (e.g., Arithmetic Intensity formulas, GPU memory hierarchies).
2.  **The Toy Simulator (`toy_engine/`)** (simplifying the concepts for step-by-step logging).
3.  **The Production Codebase (vLLM)** (referencing where these actual classes reside in the real vLLM codebase).

---

## 2. Core Codebase Mapping (Theory ➔ Toy ➔ Production)
When editing or expanding lectures, subsequent agents MUST maintain this conceptual mapping so students can bridge academia, toy simulation, and industrial-grade software engineering:

| Conceptual Concept | Toy Engine Implementation | Real vLLM Codebase File (v1) |
| :--- | :--- | :--- |
| **Virtual Memory Layout** | `toy_engine/allocator.py` (`BlockAllocator`) | `vllm/v1/core/kv_cache_manager.py` |
| **Iteration-level Scheduling** | `toy_engine/scheduler.py` (`ToyScheduler`) | `vllm/v1/core/sched/scheduler.py` |
| **Request Tracking & Queues** | `toy_engine/scheduler.py` (`RequestQueue`) | `vllm/v1/core/sched/request_queue.py` |
| **Model Forward (Prefill/Decode)**| `toy_engine/model.py` (`MockModel`) | `vllm/v1/engine/core.py` |
| **Decoupled API Server & SSE** | `toy_engine/app.py` (FastAPI Server) | `vllm/entrypoints/openai/api_server.py` |
| **Attention Backend Dispatch** | *Concept in Lesson 2.1* | `vllm/v1/attention/selector.py` |

---

## 3. Mathematical & Hardware Rigor Guidelines
Do not simplify the math or hardware concepts. Keep explanations precise:
*   **Arithmetic Intensity ($I$)**: Always formulate it as the ratio of Compute FLOPs to Memory Access Bytes ($I = \frac{\text{FLOPs}}{\text{Bytes}}$). Explain how $I$ shifts with Batch Size ($B$) and hidden dimension ($d$) using:
    $$I(B) = \frac{B}{1 + \frac{B}{d}} \text{ FLOP/Byte}$$
*   **Memory Hierarchy**: Always distinguish between **HBM (High Bandwidth Memory)** (slow, off-chip VRAM) and **SRAM (Registers, L1 Cache, Shared Memory)** (fast, on-chip).
*   **Prefill vs. Decode**: Explicitly state that Prefill is a **GEMM** operation (high arithmetic intensity, compute-bound) while Decode is a **GEMV** operation (low arithmetic intensity, memory-bound at small batch sizes).
*   **Quantization**: Quantization changes the memory denominator ($M$), effectively shifting the Roofline **Knee Point** to the left, which enables achieving compute-bound performance at lower batch sizes.

---

## 4. Repository & Workspace Constraints (Critical Gotchas)

### ⚠️ Gotcha A: React vs. Python Folder Conflict (`toy_engine/` vs. `src/`)
*   **Problem**: Docusaurus expects its React components in `src/`. Having our Python simulator in `src/` broke Docusaurus builds.
*   **Solution**: The Python engine was renamed to `toy_engine/`. 
*   **Rule**: Keep React code in `src/` and Python simulator code in `toy_engine/`. All Python imports must use absolute imports from `toy_engine.*`.

### ⚠️ Gotcha B: GitHub Pages Config Update via API (`PUT` vs. `PATCH`)
*   **Problem**: Updating the Pages source branch from `main` to `gh-pages` using `PATCH /repos/{owner}/{repo}/pages` returns a `404 Not Found` due to GitHub API legacy routing.
*   **Solution**: Use a **`PUT`** request to the same endpoint with the payload:
    ```json
    {
      "source": {
        "branch": "gh-pages",
        "path": "/"
      }
    }
    ```
    This returns a successful `HTTP 204 No Content`.

### ⚠️ Gotcha C: KaTeX CSS Subresource Integrity (SRI)
*   **Problem**: Wrong SRI hashes in `docusaurus.config.ts` block the CSS stylesheet from loading, resulting in duplicate formula rendering (raw LaTeX + MathML layout).
*   **Solution**: Use the exact correct SRI hash for KaTeX `0.16.8`:
    ```typescript
    integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn'
    ```

---

## 5. Course Roadmaps & Backlog for Future Agents
When tasked with writing new content or modifying code:
1.  **Validate Locally**: Always run `npm run build` before pushing to `main`. It must compile with **0 errors and 0 warnings**.
2.  **Maintain Sidebar Order**: Use Docusaurus decimal numbers for `sidebar_position` (e.g., `3.5` for Lesson 1.1) to avoid re-positioning existing lessons.
3.  **Feature Backlog**:
    *   [ ] Implement **Chunked Prefill** in `toy_engine/scheduler.py` by breaking down the mock prompt processing latency across multiple iterations.
    *   [ ] Expand **Lesson 5 & 6** codebase deep dives as vLLM v1 matures (specifically focusing on `vllm/v1/core/sched/request_queue.py` and `vllm/v1/engine/core.py`).
    *   [ ] Update the `toy_engine` CLI client (`client.py`) to output a visual ASCII performance chart showing throughput over time.
