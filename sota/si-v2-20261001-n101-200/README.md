# si-v2-20261001-n101-200: SI-v2 solver sweep extended to N = 101..200

**76 WIN** against the Packomania `csqv` table retrieved **2026-10-01 10:11 UTC** (`sota/packomania/history/packomania_csqv_2026-10-01.json`), out of 100 sizes searched.
Only WIN packings (strictly feasible in exact rational arithmetic, `sum_r > record + 1e-9`) are published here.

Extension campaign status at 2026-10-01 10:12 UTC: 1186 jobs aggregated from roundE0, roundE1, roundE2, roundP1.

## How this differs from `si-v2-20260927` (N <= 100)

The N <= 100 sweep was warm-started from packings our own runs had stored. Above N = 100 we had none, so every job in this
extension was warm-started from **Packomania's published coordinates** (`https://www.packomania.com/csqv/txt/csqv<N>.txt`,
12 decimals, re-inflated to strict feasibility by an exact radius LP): `nbr` jobs see the packings of the neighbouring sizes
N±1..4 and build an N-circle start from them; `self` jobs polish the published N-circle packing itself. A result that merely
re-derives the published packing (a "tie") is therefore **not ours to claim and is not exported**; the table below reports how
many sizes ended tied or below, from the sweep's own aggregate.

| status at N = 101..200 (best strict sweep result vs the live table) | sizes |
|---|---:|
| BEAT | 76 |
| tie | 24 |

Closest non-winning results (within 1e-5 of the record): 104 (-1.2e-11), 105 (-1.1e-11), 106 (-8.7e-12), 108 (-1.1e-11), 109 (-6.1e-12), 110 (+4.3e-12), 113 (-1.3e-11), 114 (-1.1e-11), 118 (-1.3e-11), 121 (+2.0e-11), 123 (-4.4e-12), 124 (-1.9e-11), 127 (-1.4e-11), 128 (-1.3e-11), 136 (-9.9e-11), 149 (-1.5e-11), 150 (-1.5e-11), 151 (-1.7e-11), 161 (-1.4e-11), 169 (-1.6e-11), 170 (-1.9e-11), 171 (-2.1e-11), 187 (-2.5e-11), 199 (-3.9e-11).

## Wins

| N | ours Σr | record | Δ (abs) | Δ (rel) | found by | record holder |
|---:|---|---|---:|---:|---|---|
| 101 | 5.291939042346 | 5.291263171788 | +6.759e-04 | +1.28e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 102 | 5.319996728926 | 5.319420509692 | +5.762e-04 | +1.08e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 103 | 5.346473029193 | 5.345851378686 | +6.217e-04 | +1.16e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 107 | 5.455694226661 | 5.454618744051 | +1.075e-03 | +1.97e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 111 | 5.555212488792 | 5.555132386885 | +8.010e-05 | +1.44e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 112 | 5.579811038411 | 5.579487231138 | +3.238e-04 | +5.80e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 115 | 5.653298101607 | 5.652307617702 | +9.905e-04 | +1.75e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 116 | 5.678137750023 | 5.676573137511 | +1.565e-03 | +2.76e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 117 | 5.702869186618 | 5.702235739011 | +6.334e-04 | +1.11e-04 | `hpolish` self seed 0 | Eckard Specht |
| 119 | 5.753185191412 | 5.752646828336 | +5.384e-04 | +9.36e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 120 | 5.777647518851 | 5.777303097445 | +3.444e-04 | +5.96e-05 | `hpolish` self seed 0 | Elian Alfonso López Preciado |
| 122 | 5.828469948565 | 5.828135840589 | +3.341e-04 | +5.73e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 125 | 5.902676475677 | 5.902296796096 | +3.797e-04 | +6.43e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 126 | 5.927023865066 | 5.926814799439 | +2.091e-04 | +3.53e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 129 | 5.999378456469 | 5.998868370219 | +5.101e-04 | +8.50e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 130 | 6.022691506144 | 6.022424077079 | +2.674e-04 | +4.44e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 131 | 6.044967052307 | 6.044773912185 | +1.931e-04 | +3.20e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 132 | 6.067352992248 | 6.067101865258 | +2.511e-04 | +4.14e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 133 | 6.089430384704 | 6.089190901004 | +2.395e-04 | +3.93e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 134 | 6.112320860066 | 6.111583027658 | +7.378e-04 | +1.21e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 135 | 6.134407303487 | 6.133616190771 | +7.911e-04 | +1.29e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 137 | 6.179294426418 | 6.179272030442 | +2.240e-05 | +3.62e-06 | `tv16ob6` nbr seed 7203 | Eckard Specht |
| 138 | 6.201797096226 | 6.201196207424 | +6.009e-04 | +9.69e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 139 | 6.224772034856 | 6.224048511183 | +7.235e-04 | +1.16e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 140 | 6.248556124146 | 6.248286289975 | +2.698e-04 | +4.32e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 141 | 6.271529393615 | 6.270968968004 | +5.604e-04 | +8.94e-05 | `w2_0906circleenha_enhancementv` nbr seed 7101 | Eckard Specht |
| 142 | 6.295209679286 | 6.293740807174 | +1.469e-03 | +2.33e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 143 | 6.319392351687 | 6.319024034482 | +3.683e-04 | +5.83e-05 | `hpolish` self seed 0 | Eckard Specht |
| 144 | 6.342599233865 | 6.341449107518 | +1.150e-03 | +1.81e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 145 | 6.365029187765 | 6.364667389585 | +3.618e-04 | +5.68e-05 | `hpolish` self seed 0 | Eckard Specht |
| 146 | 6.387741228719 | 6.386708980523 | +1.032e-03 | +1.62e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 147 | 6.410627442871 | 6.410094597843 | +5.328e-04 | +8.31e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 148 | 6.432731989088 | 6.432461590552 | +2.704e-04 | +4.20e-05 | `tv14pf6` nbr seed 7201 | Jean-René Denoual |
| 152 | 6.520443940705 | 6.520366820754 | +7.712e-05 | +1.18e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 153 | 6.541662942927 | 6.541408426141 | +2.545e-04 | +3.89e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 154 | 6.563211094217 | 6.561927143305 | +1.284e-03 | +1.96e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 155 | 6.583307096478 | 6.582091347104 | +1.216e-03 | +1.85e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 156 | 6.603620180147 | 6.602458999560 | +1.161e-03 | +1.76e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 157 | 6.623376472218 | 6.622760196297 | +6.163e-04 | +9.31e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 158 | 6.644016750914 | 6.643473474163 | +5.433e-04 | +8.18e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 159 | 6.665381293305 | 6.663891763480 | +1.490e-03 | +2.24e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 160 | 6.686422124069 | 6.686377484769 | +4.464e-05 | +6.68e-06 | `hpolish` self seed 0 | Yue Huang |
| 162 | 6.728815520983 | 6.728706099717 | +1.094e-04 | +1.63e-05 | `w2_0906circleenha_enhancementv` nbr seed 7201 | Jean-René Denoual |
| 163 | 6.750073613931 | 6.749584271556 | +4.893e-04 | +7.25e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 164 | 6.771946802778 | 6.771350627575 | +5.962e-04 | +8.80e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 165 | 6.793231550873 | 6.792830162827 | +4.014e-04 | +5.91e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 166 | 6.815149172344 | 6.814618735087 | +5.304e-04 | +7.78e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 167 | 6.837483849387 | 6.836877925708 | +6.059e-04 | +8.86e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 168 | 6.858710997365 | 6.858106347377 | +6.046e-04 | +8.82e-05 | `hpolish` self seed 0 | Eckard Specht |
| 172 | 6.943397050008 | 6.943340559805 | +5.649e-05 | +8.14e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 173 | 6.964255012795 | 6.964232150231 | +2.286e-05 | +3.28e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 174 | 6.985344577949 | 6.985262310499 | +8.227e-05 | +1.18e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 175 | 7.005012955534 | 7.004707257451 | +3.057e-04 | +4.36e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 176 | 7.024096757341 | 7.023425437597 | +6.713e-04 | +9.56e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 177 | 7.042590101131 | 7.042544930878 | +4.517e-05 | +6.41e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 178 | 7.062649343062 | 7.061720669271 | +9.287e-04 | +1.32e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 179 | 7.082173149539 | 7.080903944143 | +1.269e-03 | +1.79e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 180 | 7.101100920619 | 7.099941756488 | +1.159e-03 | +1.63e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 181 | 7.120266927405 | 7.120170554574 | +9.637e-05 | +1.35e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 182 | 7.138841236586 | 7.138696832806 | +1.444e-04 | +2.02e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 183 | 7.158914812347 | 7.158659503357 | +2.553e-04 | +3.57e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 184 | 7.178671517948 | 7.178399270760 | +2.722e-04 | +3.79e-05 | `hpolish` self seed 0 | Wilfred Heap |
| 185 | 7.198651480801 | 7.197625081950 | +1.026e-03 | +1.43e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 186 | 7.218458933071 | 7.218383296615 | +7.564e-05 | +1.05e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 188 | 7.257784502262 | 7.256806779802 | +9.777e-04 | +1.35e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 189 | 7.277955287382 | 7.277259920003 | +6.954e-04 | +9.56e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 190 | 7.298182304138 | 7.297283358464 | +8.989e-04 | +1.23e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 191 | 7.317941398677 | 7.314938403871 | +3.003e-03 | +4.11e-04 | `hpolish` self seed 0 | Wilfred Heap |
| 192 | 7.337281474317 | 7.336598450128 | +6.830e-04 | +9.31e-05 | `hpolish` self seed 0 | Jean-René Denoual |
| 193 | 7.357357788547 | 7.357305320712 | +5.247e-05 | +7.13e-06 | `hpolish` self seed 0 | Jean-René Denoual |
| 194 | 7.377183819368 | 7.376248679483 | +9.351e-04 | +1.27e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 195 | 7.397338848760 | 7.395054318099 | +2.285e-03 | +3.09e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.722e-03 | +3.67e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.168e-04 | +1.10e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 198 | 7.456737445837 | 7.454135017058 | +2.602e-03 | +3.49e-04 | `hpolish` self seed 0 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.557e-04 | +3.41e-05 | `hpolish` self seed 0 | Jean-René Denoual |

## All records beaten, N = 1..200, against the live table (2026-10-01 10:11 UTC)

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
| 111 | 5.555212488792 | 5.555132386885 | +8.0e-05 | Jean-René Denoual |
| 112 | 5.579811038411 | 5.579487231138 | +3.2e-04 | Wilfred Heap |
| 115 | 5.653298101607 | 5.652307617702 | +9.9e-04 | Jean-René Denoual |
| 116 | 5.678137750023 | 5.676573137511 | +1.6e-03 | Wilfred Heap |
| 117 | 5.702869186618 | 5.702235739011 | +6.3e-04 | Eckard Specht |
| 119 | 5.753185191412 | 5.752646828336 | +5.4e-04 | Jean-René Denoual |
| 120 | 5.777647518851 | 5.777303097445 | +3.4e-04 | Elian Alfonso López Preciado |
| 122 | 5.828469948565 | 5.828135840589 | +3.3e-04 | Wilfred Heap |
| 125 | 5.902676475677 | 5.902296796096 | +3.8e-04 | Wilfred Heap |
| 126 | 5.927023865066 | 5.926814799439 | +2.1e-04 | Jean-René Denoual |
| 129 | 5.999378456469 | 5.998868370219 | +5.1e-04 | Jean-René Denoual |
| 130 | 6.022691506144 | 6.022424077079 | +2.7e-04 | Jean-René Denoual |
| 131 | 6.044967052307 | 6.044773912185 | +1.9e-04 | Jean-René Denoual |
| 132 | 6.067352992248 | 6.067101865258 | +2.5e-04 | Jean-René Denoual |
| 133 | 6.089430384704 | 6.089190901004 | +2.4e-04 | Jean-René Denoual |
| 134 | 6.112320860066 | 6.111583027658 | +7.4e-04 | Jean-René Denoual |
| 135 | 6.134407303487 | 6.133616190771 | +7.9e-04 | Wilfred Heap |
| 137 | 6.179294426418 | 6.179272030442 | +2.2e-05 | Eckard Specht |
| 138 | 6.201797096226 | 6.201196207424 | +6.0e-04 | Jean-René Denoual |
| 139 | 6.224772034856 | 6.224048511183 | +7.2e-04 | Jean-René Denoual |
| 140 | 6.248556124146 | 6.248286289975 | +2.7e-04 | Jean-René Denoual |
| 141 | 6.271529393615 | 6.270968968004 | +5.6e-04 | Eckard Specht |
| 142 | 6.295209679286 | 6.293740807174 | +1.5e-03 | Jean-René Denoual |
| 143 | 6.319392351687 | 6.319024034482 | +3.7e-04 | Eckard Specht |
| 144 | 6.342599233865 | 6.341449107518 | +1.2e-03 | Jean-René Denoual |
| 145 | 6.365029187765 | 6.364667389585 | +3.6e-04 | Eckard Specht |
| 146 | 6.387741228719 | 6.386708980523 | +1.0e-03 | Wilfred Heap |
| 147 | 6.410627442871 | 6.410094597843 | +5.3e-04 | Jean-René Denoual |
| 148 | 6.432731989088 | 6.432461590552 | +2.7e-04 | Jean-René Denoual |
| 152 | 6.520443940705 | 6.520366820754 | +7.7e-05 | Jean-René Denoual |
| 153 | 6.541662942927 | 6.541408426141 | +2.5e-04 | Jean-René Denoual |
| 154 | 6.563211094217 | 6.561927143305 | +1.3e-03 | Jean-René Denoual |
| 155 | 6.583307096478 | 6.582091347104 | +1.2e-03 | Jean-René Denoual |
| 156 | 6.603620180147 | 6.602458999560 | +1.2e-03 | Jean-René Denoual |
| 157 | 6.623376472218 | 6.622760196297 | +6.2e-04 | Jean-René Denoual |
| 158 | 6.644016750914 | 6.643473474163 | +5.4e-04 | Jean-René Denoual |
| 159 | 6.665381293305 | 6.663891763480 | +1.5e-03 | Wilfred Heap |
| 160 | 6.686422124069 | 6.686377484769 | +4.5e-05 | Yue Huang |
| 162 | 6.728815520983 | 6.728706099717 | +1.1e-04 | Jean-René Denoual |
| 163 | 6.750073613931 | 6.749584271556 | +4.9e-04 | Wilfred Heap |
| 164 | 6.771946802778 | 6.771350627575 | +6.0e-04 | Wilfred Heap |
| 165 | 6.793231550873 | 6.792830162827 | +4.0e-04 | Jean-René Denoual |
| 166 | 6.815149172344 | 6.814618735087 | +5.3e-04 | Jean-René Denoual |
| 167 | 6.837483849387 | 6.836877925708 | +6.1e-04 | Jean-René Denoual |
| 168 | 6.858710997365 | 6.858106347377 | +6.0e-04 | Eckard Specht |
| 172 | 6.943397050008 | 6.943340559805 | +5.6e-05 | Jean-René Denoual |
| 173 | 6.964255012795 | 6.964232150231 | +2.3e-05 | Jean-René Denoual |
| 174 | 6.985344577949 | 6.985262310499 | +8.2e-05 | Jean-René Denoual |
| 175 | 7.005012955534 | 7.004707257451 | +3.1e-04 | Jean-René Denoual |
| 176 | 7.024096757341 | 7.023425437597 | +6.7e-04 | Jean-René Denoual |
| 177 | 7.042590101131 | 7.042544930878 | +4.5e-05 | Jean-René Denoual |
| 178 | 7.062649343062 | 7.061720669271 | +9.3e-04 | Jean-René Denoual |
| 179 | 7.082173149539 | 7.080903944143 | +1.3e-03 | Jean-René Denoual |
| 180 | 7.101100920619 | 7.099941756488 | +1.2e-03 | Jean-René Denoual |
| 181 | 7.120266927405 | 7.120170554574 | +9.6e-05 | Jean-René Denoual |
| 182 | 7.138841236586 | 7.138696832806 | +1.4e-04 | Wilfred Heap |
| 183 | 7.158914812347 | 7.158659503357 | +2.6e-04 | Wilfred Heap |
| 184 | 7.178671517948 | 7.178399270760 | +2.7e-04 | Wilfred Heap |
| 185 | 7.198651480801 | 7.197625081950 | +1.0e-03 | Jean-René Denoual |
| 186 | 7.218458933071 | 7.218383296615 | +7.6e-05 | Jean-René Denoual |
| 188 | 7.257784502262 | 7.256806779802 | +9.8e-04 | Jean-René Denoual |
| 189 | 7.277955287382 | 7.277259920003 | +7.0e-04 | Jean-René Denoual |
| 190 | 7.298182304138 | 7.297283358464 | +9.0e-04 | Jean-René Denoual |
| 191 | 7.317941398677 | 7.314938403871 | +3.0e-03 | Wilfred Heap |
| 192 | 7.337281474317 | 7.336598450128 | +6.8e-04 | Jean-René Denoual |
| 193 | 7.357357788547 | 7.357305320712 | +5.2e-05 | Jean-René Denoual |
| 194 | 7.377183819368 | 7.376248679483 | +9.4e-04 | Jean-René Denoual |
| 195 | 7.397338848760 | 7.395054318099 | +2.3e-03 | Jean-René Denoual |
| 196 | 7.417474619912 | 7.414752186446 | +2.7e-03 | Jean-René Denoual |
| 197 | 7.436323901330 | 7.435507140839 | +8.2e-04 | Jean-René Denoual |
| 198 | 7.456737445837 | 7.454135017058 | +2.6e-03 | Jean-René Denoual |
| 200 | 7.494071021855 | 7.493815304616 | +2.6e-04 | Jean-René Denoual |

## Files

`pck/csqv<N>.pck` (Packomania text, centred square, %.17g), `json/out<N>.json` (corner-frame sidecar), `results.csv`, `manifest.json`,
`comparison.md` (`verify_and_compare.py compare --tol 0`). Verify any file with `python verify_and_compare.py verify pck/csqv<N>.pck`.
Sweep: 1186 jobs over 4 round(s); solvers are the SI-v2 portfolio in `solver-si-v2-20260927/`.
