# DEMO_SCRIPT.md: on-screen text and narration for every scene

Revised by Stream A after the T17 sweep (Sat 21 Nov, ~23:30). Values come from project outputs; the sources of each number are listed in the table at the end (not shown on screen). Values in [square brackets] are still pending (T26 replays, or a per-run value the pipeline must read from the showcase run). Nothing else should appear on screen: no file paths, candidate numbers, "dev" notes or working labels. Narration is about 2 words per second.

**Historic replays (T26) temper the Aegean numbers:** the sweep's truth events share the library's source geometry. Replayed with their published geometry, Amorgos 1956 (Mw 7.7) and Samos 2020 (Mw 7.0) are NOT detected by ship drift (1 to 4% of days, the false-alarm level). Precise GPS detects Amorgos in 6 to 8 minutes (with the Aegean grid), but Samos only on 5 to 14% of days.

**Two numbers may still change** depending on open decisions: the held-out false-alarm rate of the AIS network at the 100 km range cap (1.6%), and all AIS detection numbers if the optional "prewhitened" AIS scoring variant is run. The pipeline must read every value from the output files, never from this text.

Total running time: about 4 minutes 50 seconds.

---

## Scene 1: Title (6 s)

**Title:** Ships as a Tsunami Network

**Subtitle:** Whether the ships already at sea could confirm a Mediterranean tsunami faster than tide gauges

**Narration:**
> "This project asks whether the ships already at sea could confirm a Mediterranean tsunami faster than tide gauges."

---

## Scene 2: The warning gap (22 s)

**Title:** How tsunamis are currently confirmed

**Labels on the map:**
- AD 365, Crete
- 1908, Messina
- 1956, Amorgos
- 2020, Samos
- Coastal tide gauges
- Deep-sea buoys (the first in the Mediterranean, 2025)
- Coastal buoy at Stromboli (2025)
- Open sea: no instruments

**Palermo panel heading:** 23 October 2026, Mw 5.5 off Alicudi

**Palermo panel labels:**
- 12:23 Tsunami alert issued
- Schools in Palermo and Trapani evacuated
- 13:22 Alert lifted once tide gauges showed no change

(Times and details only if sourced in cards.yaml; otherwise drop the times.)

**Caption:**
> Warnings are inferred from the earthquake and then confirmed at the coast, as there are very few instruments at sea.

**Narration:**
> "The Mediterranean has a long history of tsunamis. However, warnings are still inferred from the earthquake and only confirmed once the wave reaches a tide gauge, as was the case off Palermo in October, when the alert lasted about an hour."

---

## Scene 3: Previous work (18 s)

**Title:** What previous studies have shown

**Three short findings:**
- Inazu et al. (2020): the sideways drift of 13 ships tracked the currents of the 2011 Japanese tsunami (Mw 9.0), looking back after the event
- Hossen et al. (2021): in simulations of Cascadia, ship velocity alone could not recover the tsunami
- Foster and Thomas (2024): ships fitted with precise GPS could measure the height of the wave

**Box heading:** Not tested until now

**Box items:**
- A whole fleet acting as one network
- The rate of false alarms
- The Mediterranean
- A direct comparison with tide gauges

**Inset label:** The key noise correction, repeated on real Greek ship data: 51 of 58 ships improved

**Narration:**
> "Ships clearly felt Japan's tsunami in 2011, looking back after the event. However, no study has tested whether a whole fleet could confirm a smaller tsunami as it happens, without raising false alarms."

---

## Scene 4: Ships as sensors (20 s)

**Title:** How a ship measures the sea

**Labels:**
- Heading: where the ship points
- Track: where the ship actually goes
- Sideways drift: the gap between the two
- Deep water (2,766 m): 0.03 m/s
- Shelf (61 m): 0.47 m/s

**Panel heading:** The same wave, at two depths

**Caption:**
> As the water becomes shallower, the same wave pushes ships sideways far more strongly, which makes depth the deciding factor.

**Narration:**
> "Every large ship reports both where it points and where it actually goes. The gap between the two reflects how much the water is pushing it sideways, and a tsunami pushes far harder in shallow water than in deep water."

---

## Scene 5: The data (15 s)

**Title:** Real inputs, simulated events

**Left heading:** Real data
- Seafloor depth (GEBCO)
- Ship traffic (EMODnet)
- Fault geometry (EFSM20)
- Tide gauges and buoys
- Ship behaviour (Piraeus AIS)
- Noise levels (Inazu et al., 2020)

**Right heading:** Simulated
- Earthquakes
- Tsunami waves
- Ships on a given day
- Noise and weather

**Bar heading:** Where merchant ships sail, by water depth

**Bar labels:** Under 50 m: 42% (too shallow to model) · 50 to 200 m: 21% · Over 200 m: 36%

**Narration:**
> "The sea, the ships and the faults are all real. However, the tsunamis themselves are simulated, as none this large has occurred in the Mediterranean with today's traffic."

---

## Scene 6: How the simulation works (25 s)

**Title:** How the simulation works

**Five steps:**
1. Simulate every candidate earthquake once
2. Place the ships and add realistic noise
3. Clean each ship's reports
4. Compare the fleet with every candidate earthquake
5. Compare with single ships, tide gauges and buoys

**Step 1 panel labels:**
- The fault slips and the seafloor rises
- Water piles up where more flows in than out
- Water flows from high to low

**Step 3 panel labels:**
- Raw reports
- 1-minute average
- Usual drift removed
- Compared with the true current

**Caption:**
> Each earthquake is simulated over the real seafloor and recorded at every position a ship could occupy, before being compared with what the ships report.

**Narration:**
> "Each earthquake lifts the seafloor, and standard long-wave physics carries the wave across the real seabed. We then clean each ship's reports and ask which earthquake, if any, best explains what the whole fleet is seeing."

---

## Scene 7: The reshaping (15 s)

**Title:** Reshaping the problem for a GPU

**Three pairs (the obvious way, then ours):**
- One tsunami at a time → many tsunamis in one batch
- Simulate again for every test → one reusable library
- Recount every minute → keep a running total

**Tick label under each:** Gives the same answer as the simple version

**Caption:**
> As the physics is linear at sea, each earthquake only needs to be simulated once and can then be reused for every day we test.

**Narration:**
> "As the physics is linear at sea, each earthquake only needs to be simulated once and can then be reused for every day we test. This is what turns the problem into one a GPU can solve."

---

## Scene 8: A large tsunami reaches the fleet (20 s)

**Title:** A simulated Mw 8.6 earthquake on the Hellenic Arc

**Labels:**
- Minutes after the earthquake
- Sea-surface height (m)
- Ship signal relative to its noise
- Ships above their noise: [X] of [X] (from the showcase run)
- Tide gauges triggered: [X] of 61 (from the showcase run)

**Caption:**
> Ships on the shelf south of Crete feel the tsunami clearly, while ships in deep water feel very little of it.

**Narration:**
> "As the wave spreads from the Hellenic Arc, the ships on the shelf south of Crete feel it clearly, while those in deep water feel very little."

---

## Scene 9: The network confirms the tsunami, but late (20 s)

**Title:** The network confirms the tsunami, but after the tide gauges

**Left panel heading:** Where the fleet places the earthquake
- True source (Mw 8.6)
- The network's estimate, 13 km from the true source (typical)

**Right panel heading:** Evidence from the whole fleet, relative to the alarm level
- Alarm level (set so that only 1 in 100 quiet days raises an alarm)
- Confirmed after [X] minutes (showcase run; typical 84 minutes)

**Timeline heading:** Time taken to confirm the tsunami (typical day)
- Ship network: 84 min
- First tide gauge: 5.5 min
- First buoy: [X] min (showcase run)

**Caption:**
> The fleet builds enough evidence to confirm this earthquake on 79% of simulated days and places it within about 13 km, but only after about 84 minutes, long after the nearest tide gauges.

**Narration:**
> "For an earthquake this large, the fleet confirms the tsunami on four days out of five and places it close to the true source. However, this takes well over an hour, long after the tide gauges near Crete."

---

## Scene 9b: Ships with precise GPS (12 s) (NEW)

**Title:** The same ships, with precise GPS

**Labels:**
- Same earthquake, same ships
- Confirmed on 100% of simulated days
- Typical time to confirm: 1 minute
- Assumes the seafloor moves at once and GPS noise of 5 cm

**Caption:**
> Ships above the source rise with the sea surface as soon as the seafloor moves, so precise GPS heights could confirm the tsunami within minutes.

**Narration:**
> "Fitted with precise GPS, the same ships would confirm it within about a minute, because ships above the source rise with the sea surface as soon as the seafloor moves."

---

## Scene 9c: However, this is not always the case (10 s)

**Title:** However, this is not always the case

**Labels:**
- Mw 8.0, a day with no ships on the Cretan shelf
- Alarm level not reached within 2 hours

**Caption:**
> For a smaller earthquake on a day with no ships on the shelf, the fleet points to the correct area but does not gather enough evidence to raise the alarm.

**Narration:**
> "However, for a smaller earthquake on a day with no ships on the shelf, the fleet points to the correct area but never gathers enough evidence to raise the alarm."

---

## Scene 10: Is it right? (20 s)

**Title:** Checking the physics and the detector

**Left heading:** The physics
- The wave arrives when theory predicts
- No water is created or lost
- Waves grow over shallow water as expected
- The GPU gives the same answers as the CPU
- Samos 2020: 6 of 8 comparable tide gauges within a factor of 2 of the recorded wave

**Right heading:** The detector
- Planted tsunamis of Mw 8 or more found: 37% (ship drift), 100% (precise GPS)
- Typical error in location: 15 km
- Typical error in magnitude: 0.04
- False alarms on separate quiet days: 0.2 to 1.6% (target 1%)

**Caption:**
> The solver passes standard tests and is consistent with a real recorded tsunami, while the detector is tested on tsunamis whose answer we already know.

**Narration:**
> "The physics passes standard tests and is consistent with a real recorded tsunami. As well as this, the detector was tested on tsunamis we planted ourselves, and on separate quiet days."

---

## Scene 11: One ship against the network (10 s)

**Title:** Why one ship on its own is not enough

**Labels:**
- A quiet day: noise only, no tsunami
- Single-ship rule: alarm on 100% of quiet days
- Ship network: alarm on 0.2 to 1.6% of quiet days

**Caption:**
> Judging each ship on its own raises an alarm almost every day, whereas treating the fleet as one network keeps false alarms rare.

**Narration:**
> "Judging each ship on its own raises an alarm almost every day, whereas treating the fleet as one network keeps false alarms rare."

---

## Scene 12: Speed (18 s)

**Title:** What reshaping the problem makes possible

**Ladder heading:** Time to run the full experiment
- The straightforward way, on a laptop: about 3 years
- Reshaped, on a laptop: about 5.6 hours
- Reshaped, on a GPU: about 1 hour

**Small label:** Same code and same answers on the CPU and the GPU; 17 times faster on the same machine

**Caption:**
> Reshaping the problem turns years into hours, while the GPU turns hours into minutes, which allowed the whole experiment to be rerun each time an assumption was corrected.

**Narration:**
> "Done the straightforward way, this experiment would take years on a laptop. Reshaping brings it down to hours, while the GPU brings it to about an hour, which allowed us to rerun it each time we corrected an assumption."

---

## Scene 13: Where ships help (23 s)

**Title:** What ships could confirm, and how fast

**Left heading:** The smallest earthquake ships can confirm, by source
- Tabs: Calm · Moderate · Rough · Ships with precise GPS

**Right heading:** Minutes gained or lost against tide gauges and buoys, by coast

**Key numbers (calm weather):**
- Ship drift: Aegean from Mw 7.5, Cyprus Arc from Mw 8, Hellenic and Calabrian Arcs only near Mw 8.5, for earthquakes shaped like the catalogue; real Aegean faults replayed (Amorgos, Samos) were not detected
- Ship drift beats the first tide gauge in at most 12% of detections (Aegean)
- Precise GPS: from Mw 7 everywhere, and before the first tide gauge in 72 to 92% of detections

**Caption:**
> With today's reports, ships confirm only the largest earthquakes and rarely before the tide gauges, while precise GPS on the same ships would confirm much smaller ones, usually first.

**Narration:**
> "With today's position reports, ships can confirm only the largest earthquakes, and rarely before the tide gauges. Rough weather makes this worse. However, fitting the same ships with precise GPS would let them confirm earthquakes from about magnitude 7, usually before any tide gauge."

---

## Scene 14: A real earthquake (15 s)

**Title:** Replaying the 2020 Samos earthquake with today's ships

**Labels:**
- Samos, Mw 7.0
- Kos tide gauge: recorded and simulated
- Largest wave at Kos: recorded 11.9 cm, simulated about 6 cm
- Ship drift network: not confirmed (1 to 4% of simulated days, the false-alarm level)
- Ships with precise GPS: confirmed on 5 to 14% of days

**Caption:**
> The simulated wave at Kos is about half the recorded one, within the factor of 2 we set. With today's ships, an earthquake of this size would not have been confirmed at sea.

**Narration:**
> "Replaying the 2020 Samos earthquake with today's ships allows the simulation to be checked against what the Kos tide gauge actually recorded. With today's ships, an earthquake of this size would not have been confirmed at sea."

---

## Scene 15: The answer and its limits (15 s)

**Title:** The answer and its limits

**Map legend:** Ship drift adds value · Precise GPS or buoys needed

**Limits:**
1. The model is accurate offshore only, so ships in water under 50 m are ignored
2. Earthquake sources are simplified, and Samos helped set the rule for their size; real faults with other orientations (Amorgos, Samos) were much harder to detect than the catalogue suggests
3. The speed of precise GPS assumes the seafloor moves at once and that GPS noise stays at 5 cm in any weather
4. Real Greek ship data was 17 to 21% less noisy than our calm setting, which suggests the drift results are, if anything, cautious

**Next step:** Test the network on real AIS data during a future earthquake, and trial precise GPS on a few ships

**Caption:**
> Ships already at sea could confirm only the largest Mediterranean tsunamis, and usually after the tide gauges, while precise GPS on the same ships would turn them into a fast warning network.

**Narration:**
> "In conclusion, today's ships could confirm only the largest Mediterranean tsunamis, and usually after the tide gauges. However, the same ships fitted with precise GPS could confirm earthquakes from about magnitude 7 within minutes, and knowing exactly where each works is what makes the result useful."

---

## Sources slide (shown after the video or in questions, not timed)

All citations and data sources from scripts/demo/cards.yaml and data/README.md, in one list.

---

## Where each number comes from (not shown on screen)

| Scene | Value | Source |
|---|---|---|
| 3 | 51 of 58 ships | outputs/yaw_correction.md (T23) |
| 4 | 0.03 and 0.47 m/s | outputs/cleaning_steps.png (T24, Hellenic Mw 8.0, shelf 61 m and deep 2,766 m points) |
| 5 | 42 / 21 / 36% | outputs/map1.md, merchant only, whole area of interest |
| 8 to 9 | Mw 8.6, 79% of days, 84 min, 13 km, first gauge 5.5 min | outputs/sweep/*.parquet: Hellenic map2 truth events at Mw 8.6, AIS network, calm, 40 km cap (medians over detected trials) |
| 9b | 100%, 1 min | same events, GPS network |
| 10 | 6 of 8 comparable gauges | outputs/historic_samos_gauges.csv (Marmaris not comparable at this resolution) |
| 10 | 37% / 100%, 15 km, 0.04 | outputs/sweep/twin_summary.csv and parquet, twin events, calm, 40 km (AIS location and magnitude errors) |
| 10, 11 | 0.2 to 1.6%, 100% | outputs/thresholds.csv (held-out, network AIS 40 km 0.2%, 100 km 1.6%, GPS 0.6%; literal single-ship rule 100%) |
| 12 | 3 years, 5.6 h, 1 h, 17x | outputs/reshaping_ladder.json; outputs/validation/gpu_check.json |
| 13 | region thresholds, 12%, 72 to 92% | outputs/sweep/*.parquet, map2 events, calm, 40 km, network |
| 14 | 11.9 cm recorded, about 6 cm simulated (ratio 0.54) | outputs/historic_samos_gauges.csv (kos1, after the Green's-law depth correction) |
| 13, 14 | Samos and Amorgos replay detection rates | outputs/historic_replays.md (T26) |
| 15 | 17 to 21% | outputs/noise_check_piraeus.md (real / simulated calm = 0.79 to 0.83) |
