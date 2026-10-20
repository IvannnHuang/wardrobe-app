# Wardrobe Pairing App

Personal Android tablet app (Galaxy Tab S7) for capturing clothes one garment at a time, auto-cutting and tagging them on-device, and building layered outfits on my own body figure. Everything runs locally; no photos leave the tablet.

Full project plan and rules: [CLAUDE.md](CLAUDE.md)

## Layout

| Folder | Contents |
|---|---|
| `prototype/` | Python notebooks to prove cut-out and measured tags on real photos |
| `android/` | Android Studio project (Kotlin + Jetpack Compose), created in Phase 0 |
| `test-photos/` | Local test photos, **git-ignored, never committed** |

## Setup

### Tablet (Galaxy Tab S7)
1. Settings → About tablet → Software information → tap **Build number** 7 times.
2. Settings → Developer options → enable **USB debugging** (and optionally **Wireless debugging**).
3. Connect by USB-C and accept the prompt. Check with `adb devices`.

### Computer
1. Install **Android Studio** (provides the Android SDK and `adb`).
2. Install **Python 3.11+**, then:
   ```bash
   cd prototype
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   nbstripout --install   # strips notebook outputs (which can embed photos) before commits
   ```
3. Install **Claude Code** and run it from the repo root.

### Test photos
Create `test-photos/` locally (it's ignored by git):
```
test-photos/
├── body/        front.jpg  right.jpg  back.jpg  left.jpg   (base outfit only)
└── garments/
    ├── tee-white/        front.jpg  right.jpg  back.jpg  left.jpg
    ├── jacket-denim-open/
    └── ...
```

## Development phases

0. Foundations · 1. Capture · 2. Processing · 3. Review + storage · 4. Pairing · 5. Remote + polish

See CLAUDE.md for details and "done when" criteria.
