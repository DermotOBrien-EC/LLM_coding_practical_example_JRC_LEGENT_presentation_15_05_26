---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: '#ffffff'
style: |
  /* JRC C4 visual match: blue-purple title, sans-serif body, navy bullets. */
  section {
    font-family: "Arial", "Helvetica Neue", sans-serif;
    color: #000;
    font-size: 22pt;
    padding: 48px 64px 64px 64px;
  }
  h1 {
    color: #3b3b8c;
    font-size: 36pt;
    font-weight: 400;
    margin: 0 0 24px 0;
  }
  h2 { color: #3b3b8c; font-size: 26pt; font-weight: 400; }
  ul { margin: 8px 0 8px 0; padding-left: 24px; }
  ul ul { padding-left: 28px; margin-top: 6px; }
  li { margin: 6px 0; line-height: 1.35; }
  li::marker { color: #3b3b8c; }
  strong { font-weight: 700; }
  code { font-family: "Menlo", monospace; font-size: 18pt; background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }
  section::after {
    /* JRC-style page number bottom-left */
    color: #777;
    font-size: 12pt;
    bottom: 32px;
    left: 64px;
  }
  .l1 { color: #7f7f7f; }
  .l2 { color: #d97706; }   /* darker amber than the standalone — readable on white at JRC scale */
  .l3 { color: #ea580c; }
  .headline {
    display: flex; justify-content: space-between; align-items: flex-end;
    margin: 12px 16px 16px 16px;
  }
  .headline .col { text-align: center; flex: 1; }
  .headline .num { font-size: 64pt; font-weight: 700; line-height: 1; }
  .headline .lbl { font-size: 18pt; margin-top: 8px; font-weight: 600; }
  .caption { color: #444; font-size: 16pt; text-align: center; margin-top: 12px; }
  .three-col { display: flex; gap: 18px; margin-top: 6px; }
  .three-col > div { flex: 1; border: 1.5px solid #ccc; border-radius: 6px; padding: 12px 14px; background: #fafafa; }
  .three-col h3 { margin: 0 0 6px 0; font-size: 18pt; }
  .three-col .meta { color: #555; font-size: 14pt; margin-top: 8px; }
  .three-col.l1-border > div:nth-child(1) { border-color: #7f7f7f; }
  .three-col.l2-border > div:nth-child(2) { border-color: #d97706; }
  .three-col.l3-border > div:nth-child(3) { border-color: #ea580c; }
  footer { font-size: 12pt; color: #777; }
---

# Practical Use: Writing Code & Data Viz (4 min)

- **Speaker:** Dermot
- **Question:** Q4 — How can we use AI to write simple scripts and data visualisation code? What tools are best?
- **What I'll show:** the *same* AI agent (Claude Code, Opus 4.7) on the *same* forecasting task, with three different ways of asking. The error fell by a factor of three.

<div class="headline">
  <div class="col"><div class="num l1">10.76%</div><div class="lbl l1">L1 — beginner prompt</div></div>
  <div class="col"><div class="num l2">5.52%</div><div class="lbl l2">L2 — average user</div></div>
  <div class="col"><div class="num l3">3.43%</div><div class="lbl l3">L3 — research-grade prompt</div></div>
</div>

<div class="caption">MAPE on a held-out 168-hour forecast of German hourly electricity load (Open Power System Data).</div>

---

# Three users, three prompts, three workspaces

<div class="three-col l1-border l2-border l3-border">
  <div>
    <h3 class="l1">L1 — beginner</h3>
    One sentence. No project setup.
    <div class="meta">Prompt: <b>10 words</b><br/>AGENTS.md: <i>none</i><br/>Output: three baselines, no validation</div>
  </div>
  <div>
    <h3 class="l2">L2 — average user</h3>
    Names the data file, the horizon, the libraries.
    <div class="meta">Prompt: <b>46 words</b><br/>AGENTS.md: 7 lines<br/>Output: one model, no intervals</div>
  </div>
  <div>
    <h3 class="l3">L3 — research-grade</h3>
    Structured prompt (TASK / BACKGROUND / DO&nbsp;NOT). Detailed workspace context.
    <div class="meta">Prompt: <b>1,673 words</b><br/>AGENTS.md: 113 lines<br/>Output: 6-model bake-off + calibrated intervals + methods writeup</div>
  </div>
</div>

- The headline gap is not about typing more — it's about specifying every dimension that matters: <b>task, data, models, splits, metric, format, things-not-to-do.</b>

---

# Where each model fails — the diagnostic L3 forced

![h:380 center](../runs/L3/figures/04_per_day_mape.png)

- Only **LightGBM** stays uniformly low across the week; 80%/95% intervals are calibrated at **79.8%** / **92.9%**. This per-day breakdown does not exist for L1 or L2 — they were never asked for it.

---

# Four habits that worked here — and on the next slides

1. **Specify what success looks like** — not just what to build (the metric, the format, the splits).
2. **Use a DO&nbsp;NOT section** — the cheapest, highest-leverage block of any prompt.
3. **Put standing context in `AGENTS.md`** — write it once; every future session reads it.
4. **Make the AI verify itself** — a held-out split, intervals or sensitivity checks, a written discussion of failure modes.

- Same four habits apply on the next three slides (drafting reports, literature reviews, daily co-pilot use).
- Repo: <code>github.com/DermotOBrien-EC/LLM_coding_example_jrc_andres</code>
