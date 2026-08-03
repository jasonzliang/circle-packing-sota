# submission

How to submit an improved packing to Packomania — **only for a confirmed, independently-verified win**
(currently that is **N=27**; see [`../ours-sota/`](../ours-sota/)).

- **Maintainer / contact:** Dr. Eckard Specht, Otto-von-Guericke-Universität Magdeburg —
  `eckard.specht@physik.uni-magdeburg.de` (alias `eckard.specht@ovgu.de`). Plain email; no web portal.
- **What to send:** the `.pck` file(s) for the beaten N (from `../ours-sota/wins/`), zipped if several.
  Packomania format: largest radius on line 1, author on line 2, then `x y r` per circle sorted by
  increasing radius.
- **Before sending:** verify the record is still current, confirm the `.pck` coordinate convention matches
  a file downloaded from the `csqv` table, and re-verify feasibility with
  `python3 ../solver/verify_pck.py ../ours-sota/wins/csqv27.pck --record 2.685350025228`.

**The actual drafted email (`email_draft.md`) is git-ignored on purpose** — it carries personal contact
details and is a local working file, not part of the published repo. Fill in your name/contact and send it
yourself.
