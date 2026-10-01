# si-v2-20261001-n201-300: SI-v2 solver sweep extended to N = 201..300

**86 WIN** against the Packomania `csqv` table retrieved **2026-10-01 18:39 UTC** (`sota/packomania/history/packomania_csqv_2026-10-01.json`), out of 100 sizes searched.
Only WIN packings (strictly feasible in exact rational arithmetic, `sum_r > record + 1e-9`) are published here.

Extension campaign status at 2026-10-01 19:02 UTC: 4126 jobs aggregated from roundE0, roundE1, roundE2, roundE3, roundE4, roundE5, roundE6, roundP1.

## How this differs from `si-v2-20260927` (N <= 100)

The N <= 100 sweep was warm-started from packings our own runs had stored. Above N = 100 we had none, so every job in this
extension was warm-started from **Packomania's published coordinates** (`https://www.packomania.com/csqv/txt/csqv<N>.txt`,
12 decimals, re-inflated to strict feasibility by an exact radius LP): `nbr` jobs see the packings of the neighbouring sizes
N±1..4 and build an N-circle start from them; `self` jobs polish the published N-circle packing itself. A result that merely
re-derives the published packing (a "tie") is therefore **not ours to claim and is not exported**; the table below reports how
many sizes ended tied or below, from the sweep's own aggregate.

| status at N = 201..300 (best strict sweep result vs the live table) | sizes |
|---|---:|
| BEAT | 86 |
| tie | 14 |

Closest non-winning results (within 1e-5 of the record): 203 (-2.5e-11), 204 (-6.7e-11), 207 (-2.4e-11), 227 (-5.4e-11), 233 (-2.4e-11), 241 (-3.5e-11), 260 (-5.1e-11), 261 (-3.4e-11), 266 (-3.4e-11), 267 (-7.8e-11), 268 (-1.5e-13), 274 (-4.2e-11), 297 (-5.5e-11), 300 (-8.9e-11).

## Wins

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by | record holder |
|---:|---|---|---:|---:|---|---|
| 201 | 7.513867997543 | 7.513577601745 | +2.904e-04 | +3.86e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 202 | 7.532691187047 | 7.530737482199 | +1.954e-03 | +2.59e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 205 | 7.585362715539 | 7.585145048634 | +2.177e-04 | +2.87e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 206 | 7.603841369906 | 7.602847648701 | +9.937e-04 | +1.31e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 208 | 7.639765308014 | 7.638874318313 | +8.910e-04 | +1.17e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 209 | 7.657698118407 | 7.656781531190 | +9.166e-04 | +1.20e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 210 | 7.675654097662 | 7.674608739197 | +1.045e-03 | +1.36e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 211 | 7.694453725207 | 7.693286838327 | +1.167e-03 | +1.52e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 212 | 7.712204253418 | 7.712135369769 | +6.888e-05 | +8.93e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 213 | 7.731052088427 | 7.730352006424 | +7.001e-04 | +9.06e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 214 | 7.750016651895 | 7.749920828063 | +9.582e-05 | +1.24e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 215 | 7.768732348582 | 7.768139491031 | +5.929e-04 | +7.63e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 216 | 7.787633520601 | 7.787370320365 | +2.632e-04 | +3.38e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 217 | 7.806394277851 | 7.805323305341 | +1.071e-03 | +1.37e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 218 | 7.825228062144 | 7.824384394934 | +8.437e-04 | +1.08e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 219 | 7.843875392941 | 7.843270863590 | +6.045e-04 | +7.71e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 220 | 7.863610857273 | 7.862978781947 | +6.321e-04 | +8.04e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 221 | 7.882258332009 | 7.881322193729 | +9.361e-04 | +1.19e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 222 | 7.900773652117 | 7.900508311326 | +2.653e-04 | +3.36e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 223 | 7.919578708770 | 7.918453776494 | +1.125e-03 | +1.42e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 224 | 7.937492424410 | 7.936705752698 | +7.867e-04 | +9.91e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 225 | 7.955823171940 | 7.955497269730 | +3.259e-04 | +4.10e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 226 | 7.973981065909 | 7.973731098060 | +2.500e-04 | +3.13e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 228 | 8.008374309492 | 8.008136439065 | +2.379e-04 | +2.97e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 229 | 8.026684796055 | 8.025414113644 | +1.271e-03 | +1.58e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 230 | 8.043355500284 | 8.042086155658 | +1.269e-03 | +1.58e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 231 | 8.060148954445 | 8.059954604485 | +1.943e-04 | +2.41e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 232 | 8.076147059765 | 8.075645300615 | +5.018e-04 | +6.21e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 234 | 8.110211636931 | 8.109403544494 | +8.081e-04 | +9.96e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 235 | 8.127280870897 | 8.124965979221 | +2.315e-03 | +2.85e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 236 | 8.143218415657 | 8.142183830581 | +1.035e-03 | +1.27e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 237 | 8.159675455694 | 8.159361330801 | +3.141e-04 | +3.85e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 238 | 8.177228515181 | 8.176808422664 | +4.201e-04 | +5.14e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 239 | 8.194709240524 | 8.193958609549 | +7.506e-04 | +9.16e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 240 | 8.211539686082 | 8.211062781761 | +4.769e-04 | +5.81e-05 | `w2_0906circleenha_enhancementv` nbr seed 7501 | Jean-René Denoual |
| 242 | 8.248463164210 | 8.245913651626 | +2.550e-03 | +3.09e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 243 | 8.265203307861 | 8.264972382603 | +2.309e-04 | +2.79e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 244 | 8.283234687242 | 8.282696623107 | +5.381e-04 | +6.50e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 245 | 8.301242120377 | 8.300102521900 | +1.140e-03 | +1.37e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 246 | 8.319203299346 | 8.317883144176 | +1.320e-03 | +1.59e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 247 | 8.337195946938 | 8.336758433778 | +4.375e-04 | +5.25e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 248 | 8.356030683330 | 8.354624968831 | +1.406e-03 | +1.68e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 249 | 8.372725801173 | 8.372100365514 | +6.254e-04 | +7.47e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 250 | 8.390564236331 | 8.389973532838 | +5.907e-04 | +7.04e-05 | `hpolish` self seed 0 | Elian Alfonso López Preciado |
| 251 | 8.407459377815 | 8.406300148899 | +1.159e-03 | +1.38e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 252 | 8.425064313508 | 8.424518228844 | +5.461e-04 | +6.48e-05 | `w2_0906circleenha_enhancementv` nbr seed 7501 | Jean-René Denoual |
| 253 | 8.442542096727 | 8.441592362873 | +9.497e-04 | +1.13e-04 | `tv16ob6` nbr seed 7601 | Wilfred Heap |
| 254 | 8.459543379931 | 8.458872827124 | +6.706e-04 | +7.93e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 255 | 8.477130226838 | 8.477029407017 | +1.008e-04 | +1.19e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 256 | 8.493222766774 | 8.492946191754 | +2.766e-04 | +3.26e-05 | `w2_0906circleenha_enhancementv` nbr seed 7601 | Jean-René Denoual |
| 257 | 8.508918551248 | 8.508422799988 | +4.958e-04 | +5.83e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 258 | 8.525898717018 | 8.524409881740 | +1.489e-03 | +1.75e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 259 | 8.541595771139 | 8.541425245228 | +1.705e-04 | +2.00e-05 | `w2_0823circleplai_exploreplain` nbr seed 7601 | Jean-René Denoual |
| 262 | 8.589490918071 | 8.589424076094 | +6.684e-05 | +7.78e-06 | `expl1_08276armv14` nbr seed 7601 | Jean-René Denoual |
| 263 | 8.604999318471 | 8.604914868198 | +8.445e-05 | +9.81e-06 | `w2_0906circleenha_enhancementv` nbr seed 7501 | Jean-René Denoual |
| 264 | 8.621522020243 | 8.621387757251 | +1.343e-04 | +1.56e-05 | `w2_0912transcende_transcendenc` nbr seed 7501 | Jean-René Denoual |
| 265 | 8.637255515421 | 8.636677283514 | +5.782e-04 | +6.70e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 269 | 8.702568772927 | 8.702116357477 | +4.524e-04 | +5.20e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 270 | 8.717905160793 | 8.717369002184 | +5.362e-04 | +6.15e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 271 | 8.733977036961 | 8.733631185820 | +3.459e-04 | +3.96e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 272 | 8.750296657747 | 8.750101147793 | +1.955e-04 | +2.23e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 273 | 8.767283063841 | 8.766441467546 | +8.416e-04 | +9.60e-05 | `tv16ob6` nbr seed 7601 | Jean-René Denoual |
| 275 | 8.799132481698 | 8.799100032873 | +3.245e-05 | +3.69e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 276 | 8.815531602521 | 8.815248638705 | +2.830e-04 | +3.21e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 277 | 8.832904870534 | 8.831359412359 | +1.545e-03 | +1.75e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 278 | 8.849078849651 | 8.847454899623 | +1.624e-03 | +1.84e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 279 | 8.865589291170 | 8.865026989516 | +5.623e-04 | +6.34e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 280 | 8.882380761314 | 8.881441244871 | +9.395e-04 | +1.06e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 281 | 8.899953737573 | 8.897344530591 | +2.609e-03 | +2.93e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 282 | 8.915400239831 | 8.914254825361 | +1.145e-03 | +1.28e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 283 | 8.934819071507 | 8.931076985022 | +3.742e-03 | +4.19e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 284 | 8.950565872371 | 8.946990762449 | +3.575e-03 | +4.00e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 285 | 8.966364455182 | 8.963362670814 | +3.002e-03 | +3.35e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 286 | 8.983837096627 | 8.979493799189 | +4.343e-03 | +4.84e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 287 | 8.998729293911 | 8.996138387721 | +2.591e-03 | +2.88e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 288 | 9.013780846884 | 9.013518286200 | +2.626e-04 | +2.91e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 289 | 9.028532707172 | 9.027041535254 | +1.491e-03 | +1.65e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 290 | 9.043206194969 | 9.042284795006 | +9.214e-04 | +1.02e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 291 | 9.058266977833 | 9.058001856273 | +2.651e-04 | +2.93e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 292 | 9.073690339331 | 9.072594144101 | +1.096e-03 | +1.21e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 293 | 9.088869824496 | 9.088041361012 | +8.285e-04 | +9.12e-05 | `w2_0906circleenha_enhancementv` nbr seed 7501 | Jean-René Denoual |
| 294 | 9.104009751046 | 9.103186323553 | +8.234e-04 | +9.05e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 295 | 9.118490437538 | 9.116204790462 | +2.286e-03 | +2.51e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 296 | 9.131799479623 | 9.131106084861 | +6.934e-04 | +7.59e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 298 | 9.161869538266 | 9.161800772953 | +6.877e-05 | +7.51e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 299 | 9.176856508436 | 9.176674773948 | +1.817e-04 | +1.98e-05 | `hpolish` self seed 0 | Jean-René Denoual |

## All records beaten, N = 1..300, against the live table (2026-10-01 18:39 UTC)

Combined with `si-v2-20260927` (N <= 100). A row means our published packing exceeds the live Packomania value by more than 1e-9.

| N | ours Σr | live record | margin | live holder |
|---:|---|---|---:|---|
| 63 | 4.155808089352 | 4.155766324151 | +4.2e-05 | Wilfred Heap |
| 66 | 4.256749492869 | 4.255807931448 | +9.4e-04 | Wilfred Heap |
| 78 | 4.636796920045 | 4.636377420432 | +4.2e-04 | Wilfred Heap |
| 82 | 4.756048298932 | 4.755873680213 | +1.7e-04 | Wilfred Heap |
| 83 | 4.786683838811 | 4.786393807284 | +2.9e-04 | Wilfred Heap |
| 94 | 5.103121333926 | 5.102335812493 | +7.9e-04 | Wilfred Heap |
| 96 | 5.155841390333 | 5.155358373110 | +4.8e-04 | Jean-René Denoual |
| 98 | 5.209524986811 | 5.209107205724 | +4.2e-04 | Jean-René Denoual |
| 99 | 5.236903198995 | 5.236421941919 | +4.8e-04 | Wilfred Heap |
| 100 | 5.264237427968 | 5.263989520982 | +2.5e-04 | Jean-René Denoual |
| 101 | 5.291939042346 | 5.291263171788 | +6.8e-04 | Jean-René Denoual |
| 102 | 5.319996728926 | 5.319420509692 | +5.8e-04 | Jean-René Denoual |
| 103 | 5.346473029193 | 5.345851378686 | +6.2e-04 | Jean-René Denoual |
| 107 | 5.455694226661 | 5.454618744051 | +1.1e-03 | Wilfred Heap |
| 111 | 5.555334062079 | 5.555132386885 | +2.0e-04 | Jean-René Denoual |
| 112 | 5.579811038411 | 5.579487231138 | +3.2e-04 | Wilfred Heap |
| 115 | 5.653390745810 | 5.652307617702 | +1.1e-03 | Jean-René Denoual |
| 116 | 5.678137750023 | 5.676573137511 | +1.6e-03 | Wilfred Heap |
| 117 | 5.702869186618 | 5.702235739011 | +6.3e-04 | Eckard Specht |
| 118 | 5.727629674344 | 5.727559469522 | +7.0e-05 | Jean-René Denoual |
| 119 | 5.753185191412 | 5.752646828336 | +5.4e-04 | Jean-René Denoual |
| 120 | 5.777647518851 | 5.777303097445 | +3.4e-04 | Elian Alfonso López Preciado |
| 122 | 5.828469948565 | 5.828135840589 | +3.3e-04 | Wilfred Heap |
| 123 | 5.853520686624 | 5.853393864742 | +1.3e-04 | Wilfred Heap |
| 125 | 5.902749941028 | 5.902296796096 | +4.5e-04 | Wilfred Heap |
| 126 | 5.927049561373 | 5.926814799439 | +2.3e-04 | Jean-René Denoual |
| 129 | 5.999378456469 | 5.998868370219 | +5.1e-04 | Jean-René Denoual |
| 130 | 6.022691506144 | 6.022424077079 | +2.7e-04 | Jean-René Denoual |
| 131 | 6.045071828436 | 6.044773912185 | +3.0e-04 | Jean-René Denoual |
| 132 | 6.067352992248 | 6.067101865258 | +2.5e-04 | Jean-René Denoual |
| 133 | 6.089538386516 | 6.089190901004 | +3.5e-04 | Jean-René Denoual |
| 134 | 6.112320860066 | 6.111583027658 | +7.4e-04 | Jean-René Denoual |
| 135 | 6.135034215623 | 6.133616190771 | +1.4e-03 | Wilfred Heap |
| 136 | 6.157057108359 | 6.156404709652 | +6.5e-04 | Wilfred Heap |
| 137 | 6.179880388540 | 6.179272030442 | +6.1e-04 | Eckard Specht |
| 138 | 6.201826321626 | 6.201196207424 | +6.3e-04 | Jean-René Denoual |
| 139 | 6.225610568452 | 6.224048511183 | +1.6e-03 | Jean-René Denoual |
| 140 | 6.248556124146 | 6.248286289975 | +2.7e-04 | Jean-René Denoual |
| 141 | 6.271936070020 | 6.270968968004 | +9.7e-04 | Eckard Specht |
| 142 | 6.295445902290 | 6.293740807174 | +1.7e-03 | Jean-René Denoual |
| 143 | 6.319392351687 | 6.319024034482 | +3.7e-04 | Eckard Specht |
| 144 | 6.342599233865 | 6.341449107518 | +1.2e-03 | Jean-René Denoual |
| 145 | 6.365029187765 | 6.364667389585 | +3.6e-04 | Eckard Specht |
| 146 | 6.387741228719 | 6.386708980523 | +1.0e-03 | Wilfred Heap |
| 147 | 6.410627442871 | 6.410094597843 | +5.3e-04 | Jean-René Denoual |
| 148 | 6.432731989105 | 6.432461590552 | +2.7e-04 | Jean-René Denoual |
| 152 | 6.520443940705 | 6.520366820754 | +7.7e-05 | Jean-René Denoual |
| 153 | 6.541670295894 | 6.541408426141 | +2.6e-04 | Jean-René Denoual |
| 154 | 6.563211094217 | 6.561927143305 | +1.3e-03 | Jean-René Denoual |
| 155 | 6.583307096478 | 6.582091347104 | +1.2e-03 | Jean-René Denoual |
| 156 | 6.603620180147 | 6.602458999560 | +1.2e-03 | Jean-René Denoual |
| 157 | 6.623376472218 | 6.622760196297 | +6.2e-04 | Jean-René Denoual |
| 158 | 6.644016750914 | 6.643473474163 | +5.4e-04 | Jean-René Denoual |
| 159 | 6.665381293305 | 6.663891763480 | +1.5e-03 | Wilfred Heap |
| 160 | 6.686422124069 | 6.686377484769 | +4.5e-05 | Yue Huang |
| 162 | 6.728816089964 | 6.728706099717 | +1.1e-04 | Jean-René Denoual |
| 163 | 6.750073613931 | 6.749584271556 | +4.9e-04 | Wilfred Heap |
| 164 | 6.771946802778 | 6.771350627575 | +6.0e-04 | Wilfred Heap |
| 165 | 6.793231550873 | 6.792830162827 | +4.0e-04 | Jean-René Denoual |
| 166 | 6.815408144617 | 6.814618735087 | +7.9e-04 | Jean-René Denoual |
| 167 | 6.837483849387 | 6.836877925708 | +6.1e-04 | Jean-René Denoual |
| 168 | 6.858710997365 | 6.858106347377 | +6.0e-04 | Eckard Specht |
| 171 | 6.922748197450 | 6.922613061509 | +1.4e-04 | Jean-René Denoual |
| 172 | 6.943397050008 | 6.943340559805 | +5.6e-05 | Jean-René Denoual |
| 173 | 6.964255012795 | 6.964232150231 | +2.3e-05 | Jean-René Denoual |
| 174 | 6.985344577949 | 6.985262310499 | +8.2e-05 | Jean-René Denoual |
| 175 | 7.005012955534 | 7.004707257451 | +3.1e-04 | Jean-René Denoual |
| 176 | 7.024096757341 | 7.023425437597 | +6.7e-04 | Jean-René Denoual |
| 177 | 7.042590101131 | 7.042544930878 | +4.5e-05 | Jean-René Denoual |
| 178 | 7.062649343062 | 7.061720669271 | +9.3e-04 | Jean-René Denoual |
| 179 | 7.082173149539 | 7.080903944143 | +1.3e-03 | Jean-René Denoual |
| 180 | 7.101454471435 | 7.099941756488 | +1.5e-03 | Jean-René Denoual |
| 181 | 7.120266927405 | 7.120170554574 | +9.6e-05 | Jean-René Denoual |
| 182 | 7.139059937760 | 7.138696832806 | +3.6e-04 | Wilfred Heap |
| 183 | 7.158914812347 | 7.158659503357 | +2.6e-04 | Wilfred Heap |
| 184 | 7.178671517948 | 7.178399270760 | +2.7e-04 | Wilfred Heap |
| 185 | 7.198651480801 | 7.197625081950 | +1.0e-03 | Jean-René Denoual |
| 186 | 7.218458933071 | 7.218383296615 | +7.6e-05 | Jean-René Denoual |
| 187 | 7.238253386810 | 7.237822453752 | +4.3e-04 | Jean-René Denoual |
| 188 | 7.257784502262 | 7.256806779802 | +9.8e-04 | Jean-René Denoual |
| 189 | 7.277955287382 | 7.277259920003 | +7.0e-04 | Jean-René Denoual |
| 190 | 7.298182304138 | 7.297283358464 | +9.0e-04 | Jean-René Denoual |
| 191 | 7.317941398677 | 7.314938403871 | +3.0e-03 | Wilfred Heap |
| 192 | 7.338318546759 | 7.336598450128 | +1.7e-03 | Jean-René Denoual |
| 193 | 7.357900978665 | 7.357305320712 | +6.0e-04 | Jean-René Denoual |
| 194 | 7.377620865759 | 7.376248679483 | +1.4e-03 | Jean-René Denoual |
| 195 | 7.398023926483 | 7.395054318099 | +3.0e-03 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.7e-03 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.2e-04 | Jean-René Denoual |
| 198 | 7.456737445837 | 7.454135017058 | +2.6e-03 | Jean-René Denoual |
| 199 | 7.475687680419 | 7.475311776105 | +3.8e-04 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.6e-04 | Jean-René Denoual |
| 201 | 7.513867997543 | 7.513577601745 | +2.9e-04 | Jean-René Denoual |
| 202 | 7.532691187047 | 7.530737482199 | +2.0e-03 | Jean-René Denoual |
| 205 | 7.585362715539 | 7.585145048634 | +2.2e-04 | Jean-René Denoual |
| 206 | 7.603841369906 | 7.602847648701 | +9.9e-04 | Jean-René Denoual |
| 208 | 7.639765308014 | 7.638874318313 | +8.9e-04 | Jean-René Denoual |
| 209 | 7.657698118407 | 7.656781531190 | +9.2e-04 | Jean-René Denoual |
| 210 | 7.675654097662 | 7.674608739197 | +1.0e-03 | Jean-René Denoual |
| 211 | 7.694453725207 | 7.693286838327 | +1.2e-03 | Jean-René Denoual |
| 212 | 7.712204253418 | 7.712135369769 | +6.9e-05 | Jean-René Denoual |
| 213 | 7.731052088427 | 7.730352006424 | +7.0e-04 | Jean-René Denoual |
| 214 | 7.750016651895 | 7.749920828063 | +9.6e-05 | Jean-René Denoual |
| 215 | 7.768732348582 | 7.768139491031 | +5.9e-04 | Jean-René Denoual |
| 216 | 7.787633520601 | 7.787370320365 | +2.6e-04 | Jean-René Denoual |
| 217 | 7.806394277851 | 7.805323305341 | +1.1e-03 | Jean-René Denoual |
| 218 | 7.825228062144 | 7.824384394934 | +8.4e-04 | Jean-René Denoual |
| 219 | 7.843875392941 | 7.843270863590 | +6.0e-04 | Jean-René Denoual |
| 220 | 7.863610857273 | 7.862978781947 | +6.3e-04 | Wilfred Heap |
| 221 | 7.882258332009 | 7.881322193729 | +9.4e-04 | Jean-René Denoual |
| 222 | 7.900773652117 | 7.900508311326 | +2.7e-04 | Wilfred Heap |
| 223 | 7.919578708770 | 7.918453776494 | +1.1e-03 | Jean-René Denoual |
| 224 | 7.937492424410 | 7.936705752698 | +7.9e-04 | Jean-René Denoual |
| 225 | 7.955823171940 | 7.955497269730 | +3.3e-04 | Jean-René Denoual |
| 226 | 7.973981065909 | 7.973731098060 | +2.5e-04 | Jean-René Denoual |
| 228 | 8.008374309492 | 8.008136439065 | +2.4e-04 | Jean-René Denoual |
| 229 | 8.026684796055 | 8.025414113644 | +1.3e-03 | Jean-René Denoual |
| 230 | 8.043355500284 | 8.042086155658 | +1.3e-03 | Jean-René Denoual |
| 231 | 8.060148954445 | 8.059954604485 | +1.9e-04 | Jean-René Denoual |
| 232 | 8.076147059765 | 8.075645300615 | +5.0e-04 | Jean-René Denoual |
| 234 | 8.110211636931 | 8.109403544494 | +8.1e-04 | Jean-René Denoual |
| 235 | 8.127280870897 | 8.124965979221 | +2.3e-03 | Jean-René Denoual |
| 236 | 8.143218415657 | 8.142183830581 | +1.0e-03 | Jean-René Denoual |
| 237 | 8.159675455694 | 8.159361330801 | +3.1e-04 | Jean-René Denoual |
| 238 | 8.177228515181 | 8.176808422664 | +4.2e-04 | Jean-René Denoual |
| 239 | 8.194709240524 | 8.193958609549 | +7.5e-04 | Jean-René Denoual |
| 240 | 8.211539686082 | 8.211062781761 | +4.8e-04 | Jean-René Denoual |
| 242 | 8.248463164210 | 8.245913651626 | +2.5e-03 | Jean-René Denoual |
| 243 | 8.265203307861 | 8.264972382603 | +2.3e-04 | Jean-René Denoual |
| 244 | 8.283234687242 | 8.282696623107 | +5.4e-04 | Jean-René Denoual |
| 245 | 8.301242120377 | 8.300102521900 | +1.1e-03 | Jean-René Denoual |
| 246 | 8.319203299346 | 8.317883144176 | +1.3e-03 | Jean-René Denoual |
| 247 | 8.337195946938 | 8.336758433778 | +4.4e-04 | Jean-René Denoual |
| 248 | 8.356030683330 | 8.354624968831 | +1.4e-03 | Jean-René Denoual |
| 249 | 8.372725801173 | 8.372100365514 | +6.3e-04 | Jean-René Denoual |
| 250 | 8.390564236331 | 8.389973532838 | +5.9e-04 | Elian Alfonso López Preciado |
| 251 | 8.407459377815 | 8.406300148899 | +1.2e-03 | Jean-René Denoual |
| 252 | 8.425064313508 | 8.424518228844 | +5.5e-04 | Jean-René Denoual |
| 253 | 8.442542096727 | 8.441592362873 | +9.5e-04 | Wilfred Heap |
| 254 | 8.459543379931 | 8.458872827124 | +6.7e-04 | Jean-René Denoual |
| 255 | 8.477130226838 | 8.477029407017 | +1.0e-04 | Wilfred Heap |
| 256 | 8.493222766774 | 8.492946191754 | +2.8e-04 | Jean-René Denoual |
| 257 | 8.508918551248 | 8.508422799988 | +5.0e-04 | Jean-René Denoual |
| 258 | 8.525898717018 | 8.524409881740 | +1.5e-03 | Jean-René Denoual |
| 259 | 8.541595771139 | 8.541425245228 | +1.7e-04 | Jean-René Denoual |
| 262 | 8.589490918071 | 8.589424076094 | +6.7e-05 | Jean-René Denoual |
| 263 | 8.604999318471 | 8.604914868198 | +8.4e-05 | Jean-René Denoual |
| 264 | 8.621522020243 | 8.621387757251 | +1.3e-04 | Jean-René Denoual |
| 265 | 8.637255515421 | 8.636677283514 | +5.8e-04 | Jean-René Denoual |
| 269 | 8.702568772927 | 8.702116357477 | +4.5e-04 | Jean-René Denoual |
| 270 | 8.717905160793 | 8.717369002184 | +5.4e-04 | Jean-René Denoual |
| 271 | 8.733977036961 | 8.733631185820 | +3.5e-04 | Jean-René Denoual |
| 272 | 8.750296657747 | 8.750101147793 | +2.0e-04 | Jean-René Denoual |
| 273 | 8.767283063841 | 8.766441467546 | +8.4e-04 | Jean-René Denoual |
| 275 | 8.799132481698 | 8.799100032873 | +3.2e-05 | Jean-René Denoual |
| 276 | 8.815531602521 | 8.815248638705 | +2.8e-04 | Jean-René Denoual |
| 277 | 8.832904870534 | 8.831359412359 | +1.5e-03 | Jean-René Denoual |
| 278 | 8.849078849651 | 8.847454899623 | +1.6e-03 | Jean-René Denoual |
| 279 | 8.865589291170 | 8.865026989516 | +5.6e-04 | Jean-René Denoual |
| 280 | 8.882380761314 | 8.881441244871 | +9.4e-04 | Jean-René Denoual |
| 281 | 8.899953737573 | 8.897344530591 | +2.6e-03 | Jean-René Denoual |
| 282 | 8.915400239831 | 8.914254825361 | +1.1e-03 | Jean-René Denoual |
| 283 | 8.934819071507 | 8.931076985022 | +3.7e-03 | Jean-René Denoual |
| 284 | 8.950565872371 | 8.946990762449 | +3.6e-03 | Jean-René Denoual |
| 285 | 8.966364455182 | 8.963362670814 | +3.0e-03 | Jean-René Denoual |
| 286 | 8.983837096627 | 8.979493799189 | +4.3e-03 | Jean-René Denoual |
| 287 | 8.998729293911 | 8.996138387721 | +2.6e-03 | Wilfred Heap |
| 288 | 9.013780846884 | 9.013518286200 | +2.6e-04 | Wilfred Heap |
| 289 | 9.028532707172 | 9.027041535254 | +1.5e-03 | Wilfred Heap |
| 290 | 9.043206194969 | 9.042284795006 | +9.2e-04 | Jean-René Denoual |
| 291 | 9.058266977833 | 9.058001856273 | +2.7e-04 | Jean-René Denoual |
| 292 | 9.073690339331 | 9.072594144101 | +1.1e-03 | Jean-René Denoual |
| 293 | 9.088869824496 | 9.088041361012 | +8.3e-04 | Jean-René Denoual |
| 294 | 9.104009751046 | 9.103186323553 | +8.2e-04 | Jean-René Denoual |
| 295 | 9.118490437538 | 9.116204790462 | +2.3e-03 | Jean-René Denoual |
| 296 | 9.131799479623 | 9.131106084861 | +6.9e-04 | Jean-René Denoual |
| 298 | 9.161869538266 | 9.161800772953 | +6.9e-05 | Jean-René Denoual |
| 299 | 9.176856508436 | 9.176674773948 | +1.8e-04 | Jean-René Denoual |

## Files

`pck/csqv<N>.pck` (Packomania text, centred square, %.17g), `json/out<N>.json` (corner-frame sidecar), `results.csv`, `manifest.json`,
`comparison.md` (`verify_and_compare.py compare --tol 0`). Verify any file with `python verify_and_compare.py verify pck/csqv<N>.pck`.
Sweep: 4126 jobs over 8 round(s); solvers are the SI-v2 portfolio in `solver-si-v2-20260927/`.
