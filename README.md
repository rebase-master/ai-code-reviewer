# AI Pair Engineer — cross-model review loop & trust gate

An AI pair engineer for engineering teams. Give it a code snippet and it **detects a
design flaw**, **writes a failing test**, and **refactors** the code — inside a bounded
**code → review → feedback** loop where a *different reviewer model* and the test
results feed back until the change converges. A deterministic **trust gate** then
decides whether the agent may `auto-apply`, `suggest`, or `escalate-to-human`.

It ships with an **evaluation harness** that measures the system on a labeled dataset
— including the headline guarantee: **zero unsafe auto-applies**.

🚧 **Build in progress.** Shipped in small, reviewed phases — see the commit history.
Full run instructions, eval results, and screenshots arrive in the final phase.
