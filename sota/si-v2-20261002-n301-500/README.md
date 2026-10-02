# si-v2-20261002-n301-500: SI-v2 solver sweep extended to N = 301..500

**53 WIN** against the Packomania `csqv` table retrieved **2026-10-02 14:53 UTC** (`sota/packomania/history/packomania_csqv_2026-10-01.json`), out of 199 sizes posted by Packomania in this range.
Only WIN packings (strictly feasible in exact rational arithmetic, `sum_r > record + 1e-9`) are published here.

Extension campaign status at 2026-10-02 15:31 UTC: 8429 jobs aggregated from roundE0, roundE1, roundE10, roundE11, roundE2, roundE3, roundE4, roundE5, roundE6, roundE7, roundE8, roundE9, roundP1.

## How this differs from `si-v2-20260927` (N <= 100)

The N <= 100 sweep was warm-started from packings our own runs had stored. Above N = 100 we had none, so every job in this
extension was warm-started from **Packomania's published coordinates** (`https://www.packomania.com/csqv/txt/csqv<N>.txt`,
12 decimals, re-inflated to strict feasibility by an exact radius LP): `nbr` jobs see the packings of the neighbouring sizes
N±1..4 and build an N-circle start from them; `self` jobs polish the published N-circle packing itself. A result that merely
re-derives the published packing (a "tie") is therefore **not ours to claim and is not exported**; the table below reports how
many sizes ended tied or below, from the sweep's own aggregate.

| status at N = 301..500 (best strict sweep result vs the live table) | sizes |
|---|---:|
| - | 124 |
| BEAT | 53 |
| below | 17 |
| tie | 5 |

Closest non-winning results (within 1e-5 of the record): 367 (-3.7e-10), 455 (-1.3e-10), 487 (-1.4e-10), 489 (-4.9e-10), 492 (-4.9e-10).

## Wins

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by | record holder |
|---:|---|---|---:|---:|---|---|
| 301 | 9.208687464779 | 9.206706957124 | +1.981e-03 | +2.15e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 302 | 9.224733176596 | 9.222457409110 | +2.276e-03 | +2.47e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 303 | 9.238782480982 | 9.237872384633 | +9.101e-04 | +9.85e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 304 | 9.254425858674 | 9.253504120852 | +9.217e-04 | +9.96e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 305 | 9.272209086599 | 9.269547427839 | +2.662e-03 | +2.87e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 306 | 9.288950901832 | 9.284302999971 | +4.648e-03 | +5.01e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 308 | 9.320920427170 | 9.319968125993 | +9.523e-04 | +1.02e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 309 | 9.336984289826 | 9.335666509317 | +1.318e-03 | +1.41e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 310 | 9.352776319212 | 9.352370720792 | +4.056e-04 | +4.34e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 311 | 9.368940064021 | 9.367949471824 | +9.906e-04 | +1.06e-04 | `w2_0912transcende_transcendenc` nbr seed 8101 | Jean-René Denoual |
| 312 | 9.384753450913 | 9.383900680341 | +8.528e-04 | +9.09e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 313 | 9.400424345873 | 9.399575833292 | +8.485e-04 | +9.03e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 314 | 9.415455693751 | 9.414131191837 | +1.325e-03 | +1.41e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 315 | 9.430792681228 | 9.430651133292 | +1.415e-04 | +1.50e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 317 | 9.462160852928 | 9.461295946830 | +8.649e-04 | +9.14e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 318 | 9.476733744237 | 9.475501126651 | +1.233e-03 | +1.30e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 320 | 9.506483820370 | 9.506126610664 | +3.572e-04 | +3.76e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 321 | 9.521945662512 | 9.521410018032 | +5.356e-04 | +5.63e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 322 | 9.535934054383 | 9.535451604148 | +4.825e-04 | +5.06e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 323 | 9.550057569242 | 9.549618944734 | +4.386e-04 | +4.59e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 324 | 9.564266944475 | 9.563484194583 | +7.827e-04 | +8.18e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 327 | 9.605627656694 | 9.605597654827 | +3.000e-05 | +3.12e-06 | `none7` nbr seed 8101 | Jean-René Denoual |
| 330 | 9.649577696187 | 9.649407813676 | +1.699e-04 | +1.76e-05 | `tv16ob6` nbr seed 8101 | Jean-René Denoual |
| 331 | 9.665073565218 | 9.661510288207 | +3.563e-03 | +3.69e-04 | `tv16ob6` nbr seed 8102 | Jean-René Denoual |
| 332 | 9.680090466938 | 9.676557878917 | +3.533e-03 | +3.65e-04 | `tv16ob6` nbr seed 8102 | Jean-René Denoual |
| 333 | 9.694081448770 | 9.690284870774 | +3.797e-03 | +3.92e-04 | `tv14pf6` nbr seed 8101 | Jean-René Denoual |
| 335 | 9.720327462573 | 9.719504825163 | +8.226e-04 | +8.46e-05 | `tv16ob6` nbr seed 8101 | Jean-René Denoual |
| 338 | 9.767449549800 | 9.762187506496 | +5.262e-03 | +5.39e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 339 | 9.782725516609 | 9.775655735345 | +7.070e-03 | +7.23e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 341 | 9.812803976030 | 9.805007478330 | +7.796e-03 | +7.95e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 343 | 9.838730400404 | 9.833219378303 | +5.511e-03 | +5.60e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 345 | 9.868164943426 | 9.862417622270 | +5.747e-03 | +5.83e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 347 | 9.900179683532 | 9.889662892792 | +1.052e-02 | +1.06e-03 | `tv16ob6` nbr seed 8101 | Jean-René Denoual |
| 350 | 9.945539598498 | 9.940090471554 | +5.449e-03 | +5.48e-04 | `tv16ob6` nbr seed 8101 | Jean-René Denoual |
| 355 | 10.014994322374 | 10.011883344382 | +3.111e-03 | +3.11e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 370 | 10.223969847335 | 10.222766514638 | +1.203e-03 | +1.18e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 385 | 10.430051306414 | 10.424399434790 | +5.652e-03 | +5.42e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 398 | 10.605133742458 | 10.603648603343 | +1.485e-03 | +1.40e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 402 | 10.658894574576 | 10.656528036072 | +2.367e-03 | +2.22e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 403 | 10.672365066839 | 10.670267535586 | +2.098e-03 | +1.97e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 414 | 10.823352333715 | 10.823111893800 | +2.404e-04 | +2.22e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 449 | 11.273153984076 | 11.271682773930 | +1.471e-03 | +1.31e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 478 | 11.633728379913 | 11.633559016748 | +1.694e-04 | +1.46e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 488 | 11.755098207376 | 11.753818598945 | +1.280e-03 | +1.09e-04 | `tv16ob6` self seed 8101 | Jean-René Denoual |
| 490 | 11.780256537222 | 11.780098018438 | +1.585e-04 | +1.35e-05 | `tv14pf6` self seed 8101 | Jean-René Denoual |
| 491 | 11.793811762475 | 11.792921656649 | +8.901e-04 | +7.55e-05 | `tv16ob6` self seed 8101 | Jean-René Denoual |
| 494 | 11.826118343798 | 11.824004090539 | +2.114e-03 | +1.79e-04 | `tv16ob6` self seed 8101 | Jean-René Denoual |
| 495 | 11.839754102831 | 11.836128159654 | +3.626e-03 | +3.06e-04 | `tv16ob6` self seed 8101 | Jean-René Denoual |
| 496 | 11.852382017607 | 11.847856118786 | +4.526e-03 | +3.82e-04 | `tv16ob6` self seed 8101 | Jean-René Denoual |
| 497 | 11.864458301663 | 11.860375524037 | +4.083e-03 | +3.44e-04 | `tv16ob6` self seed 8101 | Jean-René Denoual |
| 498 | 11.878206095286 | 11.872093599977 | +6.112e-03 | +5.15e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 499 | 11.890330089707 | 11.883589880191 | +6.740e-03 | +5.67e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 500 | 11.903170335725 | 11.901744072026 | +1.426e-03 | +1.20e-04 | `tv16ob6` self seed 8101 | Jean-René Denoual |

## All records beaten, N = 1..10000, against the live table (2026-10-02 14:53 UTC)

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
| 201 | 7.514215471507 | 7.513577601745 | +6.4e-04 | Jean-René Denoual |
| 202 | 7.532691187047 | 7.530737482199 | +2.0e-03 | Jean-René Denoual |
| 203 | 7.550506781501 | 7.549453338312 | +1.1e-03 | Jean-René Denoual |
| 204 | 7.567417355417 | 7.567040637207 | +3.8e-04 | Jean-René Denoual |
| 205 | 7.585501515237 | 7.585145048634 | +3.6e-04 | Jean-René Denoual |
| 206 | 7.603841369906 | 7.602847648701 | +9.9e-04 | Jean-René Denoual |
| 207 | 7.622201198658 | 7.620986073417 | +1.2e-03 | Jean-René Denoual |
| 208 | 7.639765308014 | 7.638874318313 | +8.9e-04 | Jean-René Denoual |
| 209 | 7.657963657677 | 7.656781531190 | +1.2e-03 | Jean-René Denoual |
| 210 | 7.675809055326 | 7.674608739197 | +1.2e-03 | Jean-René Denoual |
| 211 | 7.694453725207 | 7.693286838327 | +1.2e-03 | Jean-René Denoual |
| 212 | 7.712204253418 | 7.712135369769 | +6.9e-05 | Jean-René Denoual |
| 213 | 7.731052088427 | 7.730352006424 | +7.0e-04 | Jean-René Denoual |
| 214 | 7.750016651895 | 7.749920828063 | +9.6e-05 | Jean-René Denoual |
| 215 | 7.768840993667 | 7.768139491031 | +7.0e-04 | Jean-René Denoual |
| 216 | 7.787633520601 | 7.787370320365 | +2.6e-04 | Jean-René Denoual |
| 217 | 7.806394277851 | 7.805323305341 | +1.1e-03 | Jean-René Denoual |
| 218 | 7.825291320960 | 7.824384394934 | +9.1e-04 | Jean-René Denoual |
| 219 | 7.843875392941 | 7.843270863590 | +6.0e-04 | Jean-René Denoual |
| 220 | 7.863610857273 | 7.862978781947 | +6.3e-04 | Wilfred Heap |
| 221 | 7.882258332009 | 7.881322193729 | +9.4e-04 | Jean-René Denoual |
| 222 | 7.900773652117 | 7.900508311326 | +2.7e-04 | Wilfred Heap |
| 223 | 7.919578708770 | 7.918453776494 | +1.1e-03 | Jean-René Denoual |
| 224 | 7.937492424410 | 7.936705752698 | +7.9e-04 | Jean-René Denoual |
| 225 | 7.955823171940 | 7.955497269730 | +3.3e-04 | Jean-René Denoual |
| 226 | 7.973981065909 | 7.973731098060 | +2.5e-04 | Jean-René Denoual |
| 227 | 7.991210921166 | 7.991139399236 | +7.2e-05 | Jean-René Denoual |
| 228 | 8.008374309492 | 8.008136439065 | +2.4e-04 | Jean-René Denoual |
| 229 | 8.026684796055 | 8.025414113644 | +1.3e-03 | Jean-René Denoual |
| 230 | 8.043355500284 | 8.042086155658 | +1.3e-03 | Jean-René Denoual |
| 231 | 8.060148954445 | 8.059954604485 | +1.9e-04 | Jean-René Denoual |
| 232 | 8.076147059765 | 8.075645300615 | +5.0e-04 | Jean-René Denoual |
| 233 | 8.094319110390 | 8.093167942343 | +1.2e-03 | Jean-René Denoual |
| 234 | 8.110211636931 | 8.109403544494 | +8.1e-04 | Jean-René Denoual |
| 235 | 8.127280870897 | 8.124965979221 | +2.3e-03 | Jean-René Denoual |
| 236 | 8.143218415657 | 8.142183830581 | +1.0e-03 | Jean-René Denoual |
| 237 | 8.160815684546 | 8.159361330801 | +1.5e-03 | Jean-René Denoual |
| 238 | 8.177228515181 | 8.176808422664 | +4.2e-04 | Jean-René Denoual |
| 239 | 8.194709240524 | 8.193958609549 | +7.5e-04 | Jean-René Denoual |
| 240 | 8.212955975303 | 8.211062781761 | +1.9e-03 | Jean-René Denoual |
| 241 | 8.230636149532 | 8.228747504774 | +1.9e-03 | Wilfred Heap |
| 242 | 8.248463164210 | 8.245913651626 | +2.5e-03 | Jean-René Denoual |
| 243 | 8.265827621895 | 8.264972382603 | +8.6e-04 | Jean-René Denoual |
| 244 | 8.284044016466 | 8.282696623107 | +1.3e-03 | Jean-René Denoual |
| 245 | 8.301242120377 | 8.300102521900 | +1.1e-03 | Jean-René Denoual |
| 246 | 8.319606109257 | 8.317883144176 | +1.7e-03 | Jean-René Denoual |
| 247 | 8.337657453498 | 8.336758433778 | +9.0e-04 | Jean-René Denoual |
| 248 | 8.356030683330 | 8.354624968831 | +1.4e-03 | Jean-René Denoual |
| 249 | 8.372725801173 | 8.372100365514 | +6.3e-04 | Jean-René Denoual |
| 250 | 8.390564236331 | 8.389973532838 | +5.9e-04 | Elian Alfonso López Preciado |
| 251 | 8.407669216587 | 8.406300148899 | +1.4e-03 | Jean-René Denoual |
| 252 | 8.425177921053 | 8.424518228844 | +6.6e-04 | Jean-René Denoual |
| 253 | 8.442542096980 | 8.441592362873 | +9.5e-04 | Wilfred Heap |
| 254 | 8.459543379931 | 8.458872827124 | +6.7e-04 | Jean-René Denoual |
| 255 | 8.477130226838 | 8.477029407017 | +1.0e-04 | Wilfred Heap |
| 256 | 8.493244330801 | 8.492946191754 | +3.0e-04 | Jean-René Denoual |
| 257 | 8.508918551248 | 8.508422799988 | +5.0e-04 | Jean-René Denoual |
| 258 | 8.525898717018 | 8.524409881740 | +1.5e-03 | Jean-René Denoual |
| 259 | 8.541771099822 | 8.541425245228 | +3.5e-04 | Jean-René Denoual |
| 260 | 8.558295242323 | 8.557871248284 | +4.2e-04 | Jean-René Denoual |
| 261 | 8.574142975709 | 8.573654744714 | +4.9e-04 | Jean-René Denoual |
| 262 | 8.589509255113 | 8.589424076094 | +8.5e-05 | Jean-René Denoual |
| 263 | 8.605274287106 | 8.604914868198 | +3.6e-04 | Jean-René Denoual |
| 264 | 8.621529016507 | 8.621387757251 | +1.4e-04 | Jean-René Denoual |
| 265 | 8.637255515421 | 8.636677283514 | +5.8e-04 | Jean-René Denoual |
| 269 | 8.702568772927 | 8.702116357477 | +4.5e-04 | Jean-René Denoual |
| 270 | 8.718081000549 | 8.717369002184 | +7.1e-04 | Jean-René Denoual |
| 271 | 8.734620044638 | 8.733631185820 | +9.9e-04 | Jean-René Denoual |
| 272 | 8.751332428368 | 8.750101147793 | +1.2e-03 | Jean-René Denoual |
| 273 | 8.767885688779 | 8.766441467546 | +1.4e-03 | Jean-René Denoual |
| 274 | 8.783646943132 | 8.782947126324 | +7.0e-04 | Jean-René Denoual |
| 275 | 8.799821274737 | 8.799100032873 | +7.2e-04 | Jean-René Denoual |
| 276 | 8.815791617727 | 8.815248638705 | +5.4e-04 | Jean-René Denoual |
| 277 | 8.833012846276 | 8.831359412359 | +1.7e-03 | Jean-René Denoual |
| 278 | 8.849078849651 | 8.847454899623 | +1.6e-03 | Jean-René Denoual |
| 279 | 8.865589291170 | 8.865026989516 | +5.6e-04 | Jean-René Denoual |
| 280 | 8.884272111120 | 8.881441244871 | +2.8e-03 | Jean-René Denoual |
| 281 | 8.901240953109 | 8.897344530591 | +3.9e-03 | Jean-René Denoual |
| 282 | 8.917797248353 | 8.914254825361 | +3.5e-03 | Jean-René Denoual |
| 283 | 8.935604389184 | 8.931076985022 | +4.5e-03 | Jean-René Denoual |
| 284 | 8.950764486072 | 8.946990762449 | +3.8e-03 | Jean-René Denoual |
| 285 | 8.967891423896 | 8.963362670814 | +4.5e-03 | Jean-René Denoual |
| 286 | 8.983837096627 | 8.979493799189 | +4.3e-03 | Jean-René Denoual |
| 287 | 8.998763554843 | 8.996138387721 | +2.6e-03 | Wilfred Heap |
| 288 | 9.013919095316 | 9.013518286200 | +4.0e-04 | Wilfred Heap |
| 289 | 9.028618602620 | 9.027041535254 | +1.6e-03 | Wilfred Heap |
| 290 | 9.043206194969 | 9.042284795006 | +9.2e-04 | Jean-René Denoual |
| 291 | 9.058616578476 | 9.058001856273 | +6.1e-04 | Jean-René Denoual |
| 292 | 9.073996180440 | 9.072594144101 | +1.4e-03 | Jean-René Denoual |
| 293 | 9.088930498053 | 9.088041361012 | +8.9e-04 | Jean-René Denoual |
| 294 | 9.104009751046 | 9.103186323553 | +8.2e-04 | Jean-René Denoual |
| 295 | 9.119378513625 | 9.116204790462 | +3.2e-03 | Jean-René Denoual |
| 296 | 9.134896514571 | 9.131106084861 | +3.8e-03 | Jean-René Denoual |
| 297 | 9.149420862049 | 9.147139780469 | +2.3e-03 | Jean-René Denoual |
| 298 | 9.164640675262 | 9.161800772953 | +2.8e-03 | Jean-René Denoual |
| 299 | 9.179904120375 | 9.176674773948 | +3.2e-03 | Jean-René Denoual |
| 300 | 9.196770845441 | 9.192717162187 | +4.1e-03 | Jean-René Denoual |
| 301 | 9.208687464779 | 9.206706957124 | +2.0e-03 | Jean-René Denoual |
| 302 | 9.224733176596 | 9.222457409110 | +2.3e-03 | Jean-René Denoual |
| 303 | 9.238782480982 | 9.237872384633 | +9.1e-04 | Jean-René Denoual |
| 304 | 9.254425858674 | 9.253504120852 | +9.2e-04 | Jean-René Denoual |
| 305 | 9.272209086599 | 9.269547427839 | +2.7e-03 | Jean-René Denoual |
| 306 | 9.288950901832 | 9.284302999971 | +4.6e-03 | Jean-René Denoual |
| 308 | 9.320920427170 | 9.319968125993 | +9.5e-04 | Jean-René Denoual |
| 309 | 9.336984289826 | 9.335666509317 | +1.3e-03 | Jean-René Denoual |
| 310 | 9.352776319212 | 9.352370720792 | +4.1e-04 | Jean-René Denoual |
| 311 | 9.368940064021 | 9.367949471824 | +9.9e-04 | Jean-René Denoual |
| 312 | 9.384753450913 | 9.383900680341 | +8.5e-04 | Jean-René Denoual |
| 313 | 9.400424345873 | 9.399575833292 | +8.5e-04 | Jean-René Denoual |
| 314 | 9.415455693751 | 9.414131191837 | +1.3e-03 | Jean-René Denoual |
| 315 | 9.430792681228 | 9.430651133292 | +1.4e-04 | Jean-René Denoual |
| 317 | 9.462160852928 | 9.461295946830 | +8.6e-04 | Jean-René Denoual |
| 318 | 9.476733744237 | 9.475501126651 | +1.2e-03 | Jean-René Denoual |
| 320 | 9.506483820370 | 9.506126610664 | +3.6e-04 | Jean-René Denoual |
| 321 | 9.521945662512 | 9.521410018032 | +5.4e-04 | Jean-René Denoual |
| 322 | 9.535934054383 | 9.535451604148 | +4.8e-04 | Jean-René Denoual |
| 323 | 9.550057569242 | 9.549618944734 | +4.4e-04 | Jean-René Denoual |
| 324 | 9.564266944475 | 9.563484194583 | +7.8e-04 | Jean-René Denoual |
| 327 | 9.605627656694 | 9.605597654827 | +3.0e-05 | Jean-René Denoual |
| 330 | 9.649577696187 | 9.649407813676 | +1.7e-04 | Jean-René Denoual |
| 331 | 9.665073565218 | 9.661510288207 | +3.6e-03 | Jean-René Denoual |
| 332 | 9.680090466938 | 9.676557878917 | +3.5e-03 | Jean-René Denoual |
| 333 | 9.694081448770 | 9.690284870774 | +3.8e-03 | Jean-René Denoual |
| 335 | 9.720327462573 | 9.719504825163 | +8.2e-04 | Jean-René Denoual |
| 338 | 9.767449549800 | 9.762187506496 | +5.3e-03 | Jean-René Denoual |
| 339 | 9.782725516609 | 9.775655735345 | +7.1e-03 | Jean-René Denoual |
| 341 | 9.812803976030 | 9.805007478330 | +7.8e-03 | Jean-René Denoual |
| 343 | 9.838730400404 | 9.833219378303 | +5.5e-03 | Jean-René Denoual |
| 345 | 9.868164943426 | 9.862417622270 | +5.7e-03 | Jean-René Denoual |
| 347 | 9.900179683532 | 9.889662892792 | +1.1e-02 | Jean-René Denoual |
| 350 | 9.945539598498 | 9.940090471554 | +5.4e-03 | Jean-René Denoual |
| 355 | 10.014994322374 | 10.011883344382 | +3.1e-03 | Jean-René Denoual |
| 370 | 10.223969847335 | 10.222766514638 | +1.2e-03 | Jean-René Denoual |
| 385 | 10.430051306414 | 10.424399434790 | +5.7e-03 | Jean-René Denoual |
| 398 | 10.605133742458 | 10.603648603343 | +1.5e-03 | Jean-René Denoual |
| 402 | 10.658894574576 | 10.656528036072 | +2.4e-03 | Jean-René Denoual |
| 403 | 10.672365066839 | 10.670267535586 | +2.1e-03 | Jean-René Denoual |
| 414 | 10.823352333715 | 10.823111893800 | +2.4e-04 | Jean-René Denoual |
| 449 | 11.273153984076 | 11.271682773930 | +1.5e-03 | Jean-René Denoual |
| 478 | 11.633728379913 | 11.633559016748 | +1.7e-04 | Jean-René Denoual |
| 488 | 11.755098207376 | 11.753818598945 | +1.3e-03 | Jean-René Denoual |
| 490 | 11.780256537222 | 11.780098018438 | +1.6e-04 | Jean-René Denoual |
| 491 | 11.793811762475 | 11.792921656649 | +8.9e-04 | Jean-René Denoual |
| 494 | 11.826118343798 | 11.824004090539 | +2.1e-03 | Jean-René Denoual |
| 495 | 11.839754102831 | 11.836128159654 | +3.6e-03 | Jean-René Denoual |
| 496 | 11.852382017607 | 11.847856118786 | +4.5e-03 | Jean-René Denoual |
| 497 | 11.864458301663 | 11.860375524037 | +4.1e-03 | Jean-René Denoual |
| 498 | 11.878206095286 | 11.872093599977 | +6.1e-03 | Jean-René Denoual |
| 499 | 11.890330089707 | 11.883589880191 | +6.7e-03 | Jean-René Denoual |
| 500 | 11.903170335725 | 11.901744072026 | +1.4e-03 | Jean-René Denoual |
| 545 | 12.434898556503 | 12.428407647582 | +6.5e-03 | Wilfred Heap |
| 576 | 12.775149186935 | 12.765618381692 | +9.5e-03 | Eckard Specht |
| 613 | 13.184368744690 | 13.178055048167 | +6.3e-03 | Wilfred Heap |
| 676 | 13.853532292832 | 13.843506930660 | +1.0e-02 | Wilfred Heap |
| 685 | 13.929349872490 | 13.915210345965 | +1.4e-02 | Zeeshan Tariq |
| 729 | 14.391974693301 | 14.380692110222 | +1.1e-02 | Wilfred Heap |
| 761 | 14.701121288267 | 14.691043947468 | +1.0e-02 | Wilfred Heap |
| 784 | 14.929313813180 | 14.925744508409 | +3.6e-03 | Wilfred Heap |
| 841 | 15.468057245963 | 15.467205852816 | +8.5e-04 | Wilfred Heap |
| 900 | 16.006738603334 | 16.003323612102 | +3.4e-03 | Wilfred Heap |
| 925 | 16.219461696123 | 16.213671971894 | +5.8e-03 | Wilfred Heap |
| 961 | 16.546730901447 | 16.546634626446 | +9.6e-05 | Wilfred Heap |

## Files

`pck/csqv<N>.pck` (Packomania text, centred square, %.17g), `json/out<N>.json` (corner-frame sidecar), `results.csv`, `manifest.json`,
`comparison.md` (`verify_and_compare.py compare --tol 0`). Verify any file with `python verify_and_compare.py verify pck/csqv<N>.pck`.
Sweep: 8429 jobs over 13 round(s); solvers are the SI-v2 portfolio in `solver-si-v2-20260927/`.
