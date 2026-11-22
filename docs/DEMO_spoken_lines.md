# Spoken lines: what the current data supports (T27 review, Sat 21 Nov ~22:30)

`docs/DEMO.md` is unchanged. These are suggested replacements where the rendered scene and its data no
longer support the draft line. Each keeps roughly 2 words per second of scene. Re-check these once
T17, T18, T25 and T26 land, because several depend on pending results.

| # | Draft line (DEMO.md) | Problem in current data | Suggested line |
|---|---|---|---|
| 8 | "The same wave pushes a shelf ship at several times its noise, and a deep-water ship at a fraction of it. Depth decides." | The two inset ships are reference points (shelf SNR 4.9, deep 0.26), not ships in today's 682-ship draw; the strongest real ship reaches 0.32σ. | "A ship on the Cretan shelf would feel five times its noise; one in deep water, a quarter. In today's traffic, the strongest ship reaches a third. Depth decides." |
| 9 | "Taken together, the fleet either confirms the tsunami here, or it does not, and we can say when." | Network score peaks at 112 against a dev threshold of 239: **not confirmed** within 2 h. The single-ship rule fires at 1 min, on noise. | "Taken together, today's fleet does not confirm even this magnitude 8 within two hours, while a single-ship rule fires on noise." |
| 10 | "…matches a real recorded tsunami. The detector found hundreds of tsunamis we planted, and rarely cried wolf." | Samos: 6 of 9 gauges within a factor of 2 **after** the shoaling correction; the model is low at all 9, and Samos helped set the uplift rule. Twin and false-alarm results are pending (T17). | "The physics passes standard tests and comes within a factor of two at six of nine Samos gauges. Detector results on planted tsunamis are still running." |
| 11 | "Watch each ship alone and you raise the alarm almost every day." | Measured: single-ship rule 100% of 500 held-out quiet days; network 3.0% (dev threshold set for 1%). | "Watch each ship alone and you raise the alarm every day; the network, three days in a hundred." |
| 12 | "Same code, same answers. The GPU is what lets us test thousands of days instead of one." | "Instead of one" overstates: the CPU matmul path already scores ~385,000 windows/hour. | "Same code, same answers. In one hour the GPU simulates thousands of tsunamis and scores millions of test windows." |
| 13 | "Here is where existing ships add warning time, how the weather erodes it, and what precise GPS would buy." | Maps 2 and 3 are pending (T17 sweep / T18). | Keep the draft once the maps land. Until then: "These maps will show where ships add warning time, and what weather and precise GPS change; the sweep is still running." |
| 14 | "Replaying Samos with today's ships, checked against what the Kos gauge actually recorded." | Supported. Adds: no ship exceeds 1σ (max 0.24σ). Network result pending (T26). | Optional: "Replaying Samos with today's ships: the model is close at Kos, but no ship feels the wave." |
| 15 | "For most of the sea the answer is no. Knowing exactly where it is yes is what makes it useful." | No "yes" region in the data yet; the conclusion map is pending. | Until Maps 2/3 land: "So far, even a magnitude 8 off Crete goes unconfirmed. Knowing where ships cannot help is itself a useful answer." |

Scenes 1 to 7 are supported as drafted.

## Numbers in DEMO.md Part 1 that the data no longer match
- "Inazu's single-ship rule applied to a fleet of around 2,000 ships": the project's traffic draws contain about
  650 to 733 merchant ships within 100 km of the coast (682 in draw 0). The ">99% on ~2,000 ships" figure came
  from a synthetic unit test.
- "whereas the network detector holds a 1% rate": measured 3.0% on 500 held-out quiet days, with a dev threshold
  (local 384-candidate subset) calibrated for 1%. Re-check with T17's `outputs/thresholds.csv` (`far_hold`).
- Depth shares in `outputs/pitch_outline.md` (55/20/24%) are the old all-types numbers. Merchant Map 1:
  42.3 / 21.5 / 36.2% for all traffic, 44.8 / 22.6 / 32.6% within 100 km.

## Things a judge may notice
- The "about 3 years" bar (speed slide) is not a fair like-for-like comparison. The earlier study (Inazu et al.,
  2020) only looked at about a dozen ships (13), so the straightforward method was fast enough for it. The 3 years
  is that method scaled up to our 1.4 million simulated tests, extrapolated from measured per-test timings and
  never actually run. The like-for-like figure is reshaped laptop against reshaped GPU: about 5.6 hours down to
  about 1 hour (5.6 times end to end).
- The quiet-day example's network score peaks at about 125, which is higher than the Mw 8 tsunami's 112.
  Scenes 9 and 11 sit close together, so be ready to explain it (noise sets the floor; this event's signal
  is mostly in deep water).
- The first "gauge trigger" at Paleochora is inside the source's initial subsidence zone, so it reflects the
  seafloor drop, not a wave arrival.
- The 17x GPU speed-up (scene 12) is on a synthetic flat grid at batch 64. On the real grid, B=64 ran out of GPU
  memory and B=32 gives 0.42 s per tsunami (scene 7).
