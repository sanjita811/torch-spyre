# Speaker Notes — SSD Kernel on Spyre (20 min)

**Audience:** VP, owners, non-technical managers, plus torch-spyre core contributors.
**Rule of thumb:** lead every slide with the plain-English point; the engineers will
get the depth from the detail, the managers from the headline. Open the deck
(`ssd_presentation.html`) in a browser, press **F** for fullscreen, arrow keys to advance.

**Timing target:** ~20 min talk + buffer for questions. Per-slide budget in brackets.
If you're running long, the compressible slides are 5, 8, and 9 — never cut 6, 7, or 11.

---

## Slide 1 — Title [0:30]
> "This is a project to bring a whole new *class* of AI model — state-space models —
> onto Spyre. I'll cover what we set out to do, how we did it, what works today, and
> where it goes next."

Set the frame: this is both a **delivered result** and a **roadmap for the platform**.
Don't read the three tags — let them sit.

## Slide 2 — Why SSMs matter [1:30]
The "why should I care" slide for the non-technical room.
> "Almost all the AI you hear about is built on Transformers. State-space models —
> Mamba-2 is the leading one — are a fast-growing alternative. Their key advantage:
> as the input gets longer, their cost grows *linearly*, where a Transformer grows with
> the *square* of the length. For long documents, long context, that's a big deal."

Business line — say it plainly:
> "For Spyre to be a serious AI platform, it has to run the architectures customers
> want — not only Transformers. This project shows it can."

## Slide 3 — Spyre is not a GPU [1:30]
Set up the difficulty *without* jargon.
> "The catch is that Spyre is a very different machine from a GPU. It organizes data
> into fixed memory blocks — we call them sticks — and it charges you every time you
> rearrange them. It has no built-in operation for some of the pieces the algorithm
> needs. And it works in half-precision, where numbers can overflow."

Then the punchline:
> "The standard Mamba-2 code is written *for GPUs*. None of it maps over directly. The
> real work was re-deriving the same math as operations Spyre can actually run — and
> discovering where the hardware pushes back."

## Slide 4 — Approach [2:00]
Walk the four boxes left to right — this is the one "how it works" slide.
> "The algorithm chops a long sequence into chunks. Four stages: mix *within* each
> chunk, summarize each chunk into a small state, carry that state *across* chunks —
> that's the memory of the model — and combine. Each stage, which the reference writes
> as one dense operation, we rebuild as a chain of matrix multiplies that Spyre supports."

For the engineers, name the techniques (bottom-left): cumsum-as-matmul, folding decay
into operands so we never build the giant intermediate, centered exponentials for fp16
safety, all fused into one kernel.

Design principle (bottom-right) — worth saying aloud, it shows maturity:
> "We deliberately kept the kernel model-agnostic. It computes the core SSM math and
> nothing else, so any state-space model can reuse it."

## Slide 5 — Testing [1:30]  *(compressible)*
The credibility slide — emphasize rigor.
> "We didn't just eyeball outputs. We run the original reference on CPU as ground truth.
> We built a *mirror* of our kernel in plain PyTorch that does the identical steps — so
> when something's off, we know instantly whether it's *our math* or *the hardware*. And
> we built a model that predicts Spyre's half-precision behavior without a compile, so we
> tune correctness in seconds instead of minutes."

Point at the tiles: fp32 math is essentially exact (4e-4); on-device we land at 0.0038,
right near the best half-precision can do. Every shape stays inside the budget.

## Slide 6 — Accomplishments [2:00]  *(do not cut)*
The "what did we deliver" slide. Slow down here.
> "Bottom line: the full kernel runs entirely on Spyre — no dropping back to CPU in the
> hot path. It's correct up to 32,000 tokens. It's fused into a single compiled kernel.
> And it's about twice as fast as our first working version."

Then gesture at the capability chips: fused, accurate, auto-tuned, streaming state,
safe fallback, reusable. Close with the headline:
> "This is the first structured-state-space model to run on Spyre at all."

## Slide 7 — Profiling [2:00]  *(do not cut — this is the intellectual core)*
The single most important technical insight, told simply.
> "When we profiled it, we found something decisive. The actual arithmetic takes about
> 74 *microseconds*. The kernel runs in about 20 *milliseconds*. So the math is under one
> percent of the time — the other 99-plus percent is just *moving data around* into the
> right layout."

Why it matters (right card) — say this explicitly, it's the strategic point:
> "This changed everything about how we optimized. Chasing compute would've been wasted
> effort. Every real gain had to come from moving *less* data — and it tells the backend
> team exactly where the leverage is."

## Slide 8 — Optimizations [1:30]  *(compressible)*
> "Every optimization attacked data movement." Run the four bullets quickly — factored
decay (~2×), shared layouts, halving a transfer, the tiling sweep (~34%).

Then the rigor point (right card) — engineers will respect this:
> "A hard lesson on this hardware: counting operations does *not* predict speed. A
> layout that looks cheaper can be slower. So we measured every change on the card, and
> we *rejected* several good-sounding ideas when the hardware disagreed — and wrote down
> the negatives so nobody re-tries them."

## Slide 9 — Chunk-size finding [1:30]  *(compressible)*
A concrete, memorable result. Let the chart do the talking.
> "Here's a clean example. GPUs use a large chunk size — 256. We swept it on the card.
> On Spyre the *smallest* chunk is nearly five times faster than the GPU default. The
> exact opposite conclusion — because we're memory-bound, and the big chunk builds a big
> intermediate."

Takeaway: hardware-specific tuning beats porting GPU defaults; we ship a measured
best-config per sequence length.

## Slide 10 — Envelope [1:30]
The headline capability chart.
> "Here's the full validated range. Correct and running from 4K up to 32K tokens, every
> length within the error budget. And the max length *doubled this cycle* — from 16K to
> 32K — partly just by re-measuring a hardware limit against the newest backend, which had
> quietly improved."

Flag the red bar: 64K is blocked — but by a *specific, identified* backend limitation,
not by our math. Segue straight into slide 11.

## Slide 11 — The wall / future work [2:00]  *(do not cut)*
This is where you turn a kernel into a platform ask — the part leadership cares about.
> "Past 32K, one tensor grows beyond the per-core memory limit. We have the fix — a
> smarter, sub-quadratic version of the cross-chunk step — and we've *proven it correct*
> on CPU. It needs one backend capability: today, reshaping a tensor into blocks corrupts
> its layout on the device. Fix that one thing and 64K-plus opens up."

Then widen it (right card):
> "More broadly, this project produced a *prioritized list* of backend capabilities —
> each with a minimal reproducer and a clear payoff. And these aren't SSM-specific: the
> same fixes help *any* memory-bound model on Spyre."

This is the slide that reframes the work as investment guidance for the platform team.

## Slide 12 — Impact & long-term [1:30]
> "So where does this go. Today: SSMs are proven on Spyre, we have a reusable tuned
> kernel, and we've handed the backend team an evidence-backed roadmap. The long-term
> goal is a *complete* Mamba-2 layer and the decode step — everything needed for real
> long-context inference on Spyre."

Walk the bottom timeline: DONE (core kernel) → full layer → decode → SSM serving.

## Slide 13 — Close [0:30]
> "To summarize: we took a GPU-only architecture, re-derived it for Spyre, validated it
> to the precision floor, and optimized it to its memory ceiling — correct and running up
> to 32K. The road to full SSM serving is clear, and the backend work it needs is already
> identified and prioritized. Thank you — happy to take questions."

---

## Anticipated Q&A

**"How does performance compare to a GPU?"**
Different question than it sounds — Spyre is memory-bound here, GPUs are too on Mamba.
The honest answer: we optimized to the *memory ceiling of the current backend*; absolute
GPU-vs-Spyre comparison needs the backend residency fixes on slide 11 to be fair. What we
*can* say: we're within a small factor of the half-precision numerical floor on accuracy,
and ~2× faster than our first version on speed.

**"Why is it slower at longer sequences?" (the envelope chart)**
The error grows slightly (still in budget); the *runtime* grows because the cross-chunk
step is quadratic in the number of chunks. That's exactly what the sub-quadratic scan on
slide 11 fixes — but it's gated on the backend reshape capability.

**"Is 0.0038 error good enough for production?"**
Yes — that's relative error at the fp16 numerical floor; it's within the tolerance these
models are trained and served at. fp32 math is exact to 4e-4; the rest is half-precision,
which is the intended serving precision.

**"What's the effort to get the backend fixes?"**
Each is scoped with a minimal reproducer. The top one (splitting-reshape layout) is a
well-defined capability in the layout pass. I'd defer sizing to the backend owners, but
we've done the work to make each ask concrete and testable.

**"Could this run other SSMs, or only Mamba-2?"**
The kernel is deliberately model-agnostic — it computes the core SSM operation only.
Other state-space models that share this structure can reuse it; model-specific pieces
(projections, gating) live in the layer above.

**"What's the single most important takeaway?"**
The kernel is memory-bound — the math is essentially free, data movement is everything.
That reframes both our optimization work and the backend roadmap around one lever:
keeping data resident and reducing layout conversions.

---

## Delivery reminders
- **Numbers to know cold:** ~74 µs compute vs ~20 ms runtime (memory-bound); 0.0038 error;
  32K max; ~2× redesign speedup; ~34% tiling win; chunk-64 ~5× faster than chunk-256.
- Don't say "restickify," "einsum," "L×L," "fp16 stick" to the whole room — the deck
  keeps those out on purpose. Use them only if a contributor asks.
- The two slides that land with leadership are **7 (memory-bound insight)** and **11
  (backend roadmap)**. Spend your energy there.
- If a contributor wants the deep detail (layouts, the scan reshape bug, the CPU oracle
  design), offer to go through it after — keep the main talk at altitude.
