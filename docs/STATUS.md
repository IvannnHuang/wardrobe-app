# Status

> **Summary:** Working on garment processing (base-outfit removal, clothes model speed, per-piece view voting). Current phase number to confirm against the phase table in CLAUDE.md.

**Updated:** 2026-11-01 (seeded from git log + CLAUDE.md)

## Current task
- Processing quality and speed. Latest commits:
  - `e331f04` Pieces: views vote (drop base outfit / single-view pieces)
  - `d0ad836` Clothes model: fp32 on CPU + body figure cache (~11 s per garment, was ~35 s)
  - `b48c6b4` Base-outfit removal: correct lighting with skin colour + down-weight lightness

## Progress
- Phases (CLAUDE.md): 0 Foundations · 1 Capture · 2 Processing · 3 Review + storage · 4 Pairing · 5 Remote + polish
- [ ] Confirm which phase is current and tick the finished ones here.

## Next step
- (to fill in at the next session)

## Open questions / blocked on
- Before Phase 5 adds `INTERNET` for the LAN remote: exclude or disable MediaPipe's `datatransport`.

## End-of-session checklist
1. Update this file.
2. Draft the commit message + file list (I commit).
3. `contrib-log` / `distill` if something was learned.

## Session log (newest first, one line each)
- 2026-11-01 — STATUS.md created.
