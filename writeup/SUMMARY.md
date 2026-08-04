# Summary: a self-improving AI agent beat a circle-packing record

*A short version of [the full research note](README.md). Cognizant AI Lab.*

**What it is.** Take a square and fit some number of circles inside it, any sizes you like, with no
overlaps and nothing poking out. Now make the total of all their radii as large as possible. The rules
fit in a sentence, but the problem is genuinely hard: there are enormously many near-equally-good
arrangements, and small rearrangements can pay off in ways no one can eyeball. It has become a public
benchmark for AI systems that write their own code, used as a showcase by Google DeepMind's AlphaEvolve
and by Sakana AI's ShinkaEvolve. The scorekeeper is [Packomania](https://www.packomania.com/), a
reference site that Dr. Eckard Specht has maintained for decades, listing the best arrangement anyone
has ever submitted for each number of circles. It plays the role the record books play in athletics.

**What record we broke.** For 27 circles, our solver found an arrangement with a sum of radii of
2.685978684198, against the previous listed record of 2.685350025228. The gain is +0.000628658970, about
+0.023%. That margin is tiny in absolute terms and completely decisive in context: the previous entry had
stood since 2011/12, credited to D. W. Cantrell on the sci.math forum. Packomania has since accepted our
configuration as the listed record, credited to Jason Liang, and notes its "remarkable D1 symmetry",
meaning it is a mirror image of itself across a diagonal. The scope is worth stating plainly. We improved
a long-standing classical entry, not one of the recent AI-generated ones. On the most-studied case, 26
circles, which AlphaEvolve made famous, our agent matches the best-known value but does not beat it.

**How we did it.** No person wrote the solver. An AI coding agent, built on Anthropic's Claude, worked in
short iterations: change the solver, test it against a fixed benchmark, record the outcome, commit to git,
repeat. Git is literally the agent's memory. We ran six such loops in parallel as a controlled experiment,
and the record came from the most exploratory variant, the one allowed to rewrite its own improvement
process. The method it arrived at is elegant and, notably, not the one the evolutionary systems use. Its
key insight is that the problem splits cleanly in two: *if you fix where the circle centres go, finding
the largest radii that fit is a linear program*, a classic problem type that can be solved exactly and
almost instantly. So the agent only ever searches over centre positions, and each candidate layout gets
its radii filled in perfectly. Around that it wrapped a nonlinear search over the centres with many random
restarts, plus a final rescaling step guaranteeing every answer it reports is strictly legal rather than
approximately legal. A check for copied implementations found none, so the agent reconstructed this from
its own knowledge rather than looking it up.

**The finding we did not expect.** Every version of the solver was saved as the agent worked, so we could
go back and test all of them. A single run of one version tells you little, since the result depends on
where its random starting layout lands, so we ran each version 50 times from different starts and asked
which of them could reach the record at all. The answer: the very first iteration, costing about $2.48,
could already reach the record-beating configuration. Roughly $12 more of self-improvement did not raise the peak at all. What it bought was
reliability, lifting the per-start success rate from 10% to 14%. That is a useful and slightly deflating
result. The raw capability came from the base model, early and cheaply; the self-improvement loop turned a
capable first idea into something dependable.

**Why it matters to industry.** Most AI agents today retrieve, summarize and recombine what is already
known. This is a small, checkable instance of an agent producing something that was not in the answer key.
Three things follow. First, a great deal of enterprise optimization has exactly the shape of circle
packing: maximize value subject to hard constraints that cannot be bent. Logistics and routing, chip and
board layout, warehouse slotting, scheduling. An agent that can autonomously assemble a strong custom
solver for that shape of problem is directly useful. Second, and more important, the output is verifiable.
The packing is a fixed list of coordinates that an independent checker, sharing no code with the solver,
confirms in under a second. In a field full of AI claims nobody can audit, "here is the answer and here is
an independent check" is what makes a result safe to act on. Third, the cost analysis points somewhere
specific: spend on the scaffolding that makes a capable model's first good idea reliable and transferable,
not on ever-larger blind search.

**Caveats.** The solver is not uniformly strong. On large boards it trails the records by up to about
2.3%, and at 97 circles a single run failed to return a valid packing at all. Each of the six experimental
arms is one run, so differences between them are directional rather than statistically established. The
search is random, so re-running does not guarantee re-finding a given win, though the saved configuration
stays valid forever. This is one specific verified win, not across-the-board superiority.
