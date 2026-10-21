# CLAUDE.md — Wardrobe Pairing App

Guidance for Claude Code working in this repo. Read this before every task.

## What this project is

A personal Android tablet app that:
1. Captures my clothes **one garment at a time** (4 angles: front, right, back, left) with the tablet's rear camera on a tripod.
2. Processes captures **in a background batch after the session**: cut-out → tagging → matching.
3. Lets me **review and verify** every item before it enters the wardrobe.
4. Lets me build **layered outfits on my own body figure** by swiping.

Single user (me), single device. Draft 1 = everything below. Draft 2 = auto-pairing (see end).

## Hard requirements (do not violate)

- **Everything runs locally on the tablet.** No cloud APIs, no analytics, no remote model downloads.
  - Bundle all models in the APK (MediaPipe Tasks / TensorFlow Lite). **Do not use Google ML Kit models that download through Play Services.**
  - Do not add the `INTERNET` permission for anything except the local phone-remote server (LAN only).
  - MediaPipe pulls in Google `datatransport` (usage stats), which adds `INTERNET` + `ACCESS_NETWORK_STATE`. The app manifest removes both with `tools:node="remove"`. After adding any dependency, check the merged manifest for new permissions. When Phase 5 adds `INTERNET` for the LAN remote, exclude or disable `datatransport` first.
  - Models are fetched once by `scripts/fetch-models.sh` (pinned sha256) into `android/app/src/main/assets/models/` and `prototype/models/`; they are git-ignored.
- **Photos are stored in app-private storage** (`context.filesDir`), never MediaStore / Gallery / shared storage.
- **Nothing is auto-saved to the wardrobe.** Captures go to the review tray; only my approval creates or updates an Item/Version.
- **My verified tags are never overwritten** by re-running tagging (store `source = user | model` per tag).
- **Never commit photos** (anything under `test-photos/`, `*.jpg`, `*.png` outside `android/app/src/main/res/`).

## Hardware and capture setup

- Device: **Samsung Galaxy Tab S7+** (SM-T970, Snapdragon 865+, 13MP rear AF + 5MP ultra-wide, 8MP front). minSdk 29 (shipped Android 10); currently runs Android 13 — test on that.
- Use the **13MP rear main camera**, portrait, tripod at about chest height, fixed position.
- Fixed LED lighting, plain light wall, tape floor mark.
- **Base outfit**: fitted, one solid color I don't own clothes in (e.g. muted teal). Every garment is captured over it.
- Optional: my phone as remote (browser page served by the tablet), Bluetooth shutter (volume-key events).
- Do not use the emulator for camera or ML work; test on the real tablet.

## Repo layout

```
wardrobe-app/
├── CLAUDE.md          this file
├── README.md
├── prototype/         Python notebooks: prove cut-out and measured tags first
├── android/           Android Studio project (Kotlin + Compose)
└── test-photos/       LOCAL ONLY, git-ignored
```

## Tech stack

- **Android:** Kotlin, Jetpack Compose, Material 3, single-activity + Navigation Compose
- **Camera:** CameraX (lock AE/AWB/AF per session via Camera2 interop)
- **ML (bundled):** MediaPipe Tasks: Pose Landmarker, Image Segmenter (person mask). Mobile CLIP-type image model in TFLite for semantic tags.
- **Image processing:** OpenCV for Android (or plain Kotlin/RenderScript replacement where simpler)
- **Storage:** Room (SQLite) + files in `filesDir`
- **Background work:** WorkManager (batch processing, resumable)
- **Phone remote:** Ktor embedded server, LAN only, pairing code in QR
- **Voice:** Android TextToSpeech
- **DI:** Hilt. **Images in UI:** Coil.
- **Prototype:** Python 3.11+, MediaPipe, OpenCV, NumPy, Jupyter

## Architecture

```
ui/        pairing/  wardrobe/  camera/  review/   (Compose screens + ViewModels)
domain/    layering rules, tag logic, matching, models (pure Kotlin, unit-tested)
data/      Room entities/DAOs, file store, repositories
capture/   CameraX, pose check, quality check, session state machine
pipeline/  cut-out, tagging, matching workers (WorkManager)
remote/    Ktor server + phone web page
```

Keep `domain/` free of Android dependencies so rules and tag math are unit-testable.

## Capture flow (photos only, no processing)

1. One-time **Body figure** setup: base outfit only, 4 angles, plus my height. Saves the figure + body points.
2. **Start session** → put on one garment → voice-guided front → right → back → left.
3. Pose detection checks: on mark, facing direction, fully in frame, still → beep → auto-capture.
4. **Quick quality check** per shot (<1s): on mark, in frame, sharp. Fail → redo that angle now.
5. Optional: category hint, fabric **close-up**.
6. **Next garment** → repeat. **Finish session** → enqueue batch processing.

Raw photos are written to disk before anything else so nothing is lost.

## Batch processing (after the session)

Per garment, resumable WorkManager job:
1. **Align** capture to body figure using pose landmarks.
2. **Cut-out:** person mask → remove background, base-outfit color, and body (diff vs. body figure) → edge cleanup. Open garments get transparent interiors.
3. **Tagging** (below), each tag with a confidence score.
4. **Matching:** suggest "new item" or "version of existing item".
5. Status → `READY_FOR_REVIEW`; notify "N items ready to review".

## Review screen

One garment at a time: 4 cut-out views (toggle original), preview on body figure, match choice (new / version with a label like tucked, zipped, open), tag chips with low-confidence ⚠ first.
Actions: Approve & next, Skip, Discard, Re-shoot angle, Fix cut-out (brush add/erase).
Shortcuts: Approve all confident, Apply correction to similar items.
Every tag change is saved as a `TagCorrection`.

Capture status: `CAPTURED → PROCESSING → READY_FOR_REVIEW → (NEEDS_RESHOOT) → APPROVED | DISCARDED`

## Tags

Only show tags relevant to the item's category.

| Group | Values | Method |
|---|---|---|
| Category / rank | tee, shirt, sweater, jacket, dress, pants… → rank | image model + my input |
| Color | top colors with share, name, brightness, saturation, warm/cool | pixel clustering (LAB) |
| Pattern | solid, stripe, plaid, check, floral, graphic/logo | image model |
| Style | casual, smart casual, business, formal, sporty, streetwear, minimal, preppy, vintage, outdoor | image model (multi-label + confidence) |
| Occasion | everyday, work, interview, date, party, formal event, gym, outdoor, travel, lounge | image model + defaults |
| Formality | 1–5 | category default + style |
| Warmth / season | warmth 1–5, seasons, weatherproof | category + thickness + material |
| Coverage % | torso, upperArms, forearms, thighs, knees, lowerLegs, feet, head | garment mask vs body regions from landmarks |
| Lengths | sleeve, top length, bottoms length | derived from coverage |
| Fit / neckline | slim…oversized; crew, V, collar, hood, turtleneck | outline gap + image model |
| Thickness | 1–5 | garment-to-body gap (mainly side views) + material |
| Rise (bottoms) | low / mid / high | waistband height vs hip + estimated waist |
| Leg shape (bottoms) | skinny, straight, tapered, wide, flare | leg width at thigh/knee/ankle vs my legs |
| Sheerness | opaque / semi-sheer / sheer | base-outfit color bleeding through garment pixels |
| Texture | matte, shiny, knit, fuzzy, smooth, rugged | specular highlights + texture analysis + image model (+ close-up) |
| Material | cotton, denim, knit/wool, linen, leather, fleece, synthetic, silk | image model (+ close-up) |
| Comfort | stretch, breathability 1–5, comfort 1–5 | material estimate + my input + learned from wear |
| Personal | favorite, condition, availability, timesWorn, lastWorn | me + auto-tracked |

## Layering rules

| Rank | Group | Examples |
|---|---|---|
| U1 | Inner | tank, camisole, undershirt |
| U2 | Shirt | tee, polo, button-up, blouse, **dress (top)** |
| U3 | Mid | sweater, hoodie, sweatshirt, cardigan, sweater vest |
| U4 | Light outer | blazer, denim jacket, overshirt |
| U5 | Heavy outer | coat, parka, puffer |
| L1 | Legwear | tights, leggings |
| L2 | Bottoms | pants, jeans, shorts, skirt, **dress (bottom)** |
| F | Feet | shoes, boots |
| H | Head | hats |

1. One item per rank.
2. Items auto-place by rank; I never drag layer order.
3. Jackets/coats always outermost; sweaters always above shirts.
4. Dress fills U2 + L2 (clears shirt and bottoms); U1, U3–U5, L1 still allowed.
5. L2 holds one item; L1 can go under skirt/shorts/dress.
6. Only U1–U2 can be tucked.
7. Pants cover shoes by default; boots can be set to go over pants.
8. Zipped outer layer dims (not locks) the layers below.
9. "Free order" toggle allows exceptions.
10. Rules live in an **editable config table**, not hard-coded.

**Draw order (back → front):** Body → L1 → U1/U2 if tucked → L2 → U1/U2 if untucked → U3 → U4 → U5 → shoes (or boots over pants) → hat

**Alignment:** scale/translate each garment onto the body figure via matching landmarks (tops: shoulders + hips; bottoms: hips → ankles; shoes: ankles).

## The three pages

1. **Pairing:** body figure, 4 swipe areas (head / upper / lower / shoes). Upper area has a layer stack of chips (tap to select, swipe to change, + to add; swipe only shows items valid for that rank). Long-press switches version. Rotate front/right/back/left. Save outfit, Wearing this today. Short hint when a rule changes something.
2. **Wardrobe:** grouped by category, front-view thumbnails, filter by any tag. Item detail: 4 views, versions, editable tags, outfits with this item, re-shoot/add version, delete.
3. **Camera:** live preview with pose outline and floor-mark target; start / next garment / finish / close-up; review tray badge; phone remote QR.

## Data model (Room)

```
BodyFigure     id, views(4 paths), landmarks per view, heightCm
Item           id, name, category, rank, defaultVersionId, availability, createdAt
Version        id, itemId, label, front/back/left/right paths, closeUpPath?, landmarks, capturedAt
Tag            itemId, key, value(JSON), confidence, source(user|model), updatedAt
CaptureEntry   id, sessionId, rawPaths, status, cutoutPaths, suggestedTags(JSON), suggestedMatches(JSON)
Outfit         id, name, createdAt
OutfitItem     outfitId, itemId, versionId, stateFlags
WearLog        outfitId, date, comfortRating?
TagCorrection  itemId, key, oldValue, newValue, at
LayerRule      category, rank, flags(JSON)
```

## Development phases

| Phase | Goal | Done when |
|---|---|---|
| 0. Foundations | Project skeleton, 3-page nav, Room schema, bundled models running, physical setup | App runs on tablet; models run on a test photo |
| 1. Capture | Body figure setup, session flow, voice, pose auto-capture, quality check, close-up | Several garments captured in one session, hands-free |
| 2. Processing | WorkManager batch; alignment; cut-out; color, coverage, thickness, rise, leg shape, sheerness | Test set produces clean cut-outs with correct measured tags |
| 3. Review + storage | Image-model tags, matching, review screen, touch-up, versions, wardrobe page, re-run tagging | A batch goes from capture to approved wardrobe in-app |
| 4. Pairing | Rules config, layered drawing, alignment, layer stack, swiping, versions, rotate, save/wear | All rule cases draw correctly; smooth on Tab S7+ |
| 5. Remote + polish | Ktor phone remote, BT shutter, comfort prompts, learning from corrections, local backup, firewall check | Daily use with my real wardrobe |

**Phase-zero rule:** prove cut-out and measured tags in `prototype/` (Python) on real test photos **before** porting to Kotlin. If results are poor, fix the physical setup (lighting, base color) first.

## How to work in this repo

- Work in **small tasks** scoped to a phase step (e.g. "Phase 1.3: direction check + auto-capture"). Ask before expanding scope.
- Build and install on the tablet with Gradle (`./gradlew installDebug` from `android/`) when a device is connected via `adb`.
- Write unit tests for `domain/` (layering rules, tag math, matching). Keep a regression test set of photos in `test-photos/` (local) and re-run the pipeline on it after pipeline changes.
- Commit after each working step with a clear message. Never commit photos, keystores, or `local.properties`.
- Do not add dependencies that need network access at runtime. Flag any new permission in the summary.
- Keep UI simple; performance matters more than animation polish.

## Draft 2 (not now): auto-pairing

Request like "work, rainy, 45°F, smart casual, knees covered":
1. **Filter:** availability, occasion, coverage constraints, total warmth vs temperature, thickness fit for layering, sheer → needs inner layer.
2. **Score:** color harmony, formality spread ≤ ~1, style overlap, texture mix, comfort, favorites/liked combos, not worn recently.
Show top 3–5 on the pairing page; I can swipe to adjust. Weather (if added) sends only location; never photos.
