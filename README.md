# fpv-vision

AI object detection on drone and handheld video, built as a drone/FPV video-analytics demo. People and
vehicles are boxed, tracked and counted. You can ask a vision-language model to describe the
scene, and send a SALUTE spot report to ATAK and over the Meshtastic mesh.

Supported machines:

- **Apple Silicon Mac (primary).** The Drone-ops Mac (M4 Pro) runs YOLO26s on the GPU (MPS) in
  real time, including on 4K DJI footage.
- **Raspberry Pi 5.** The same tool runs YOLO26n on the Pi's CPU (NCNN) at a lower frame rate.
  The Pi stays the hub: OpenTAKServer, MQTT/mesh bridge and the launcher.

| mode | what it is | badge on video |
|---|---|---|
| **Replay as live** (headline) | Plays your real DJI O3/O4 MP4s through the detector at real-time speed, timed to the wall clock (frames are dropped rather than slowed if we fall behind). Supports playlists, looping, a scrub bar and pause. | `REPLAY <file> mm:ss` |
| **Live camera** | Mac built-in camera, **iPhone through Continuity Camera**, or a USB capture card. Cameras are picked by name. | `LIVE` |
| **Screen mirror** (best effort) | DJI Goggles N3 → phone (DJI Fly) → mirrored to the Mac → screen capture with a crop. | `SCREEN MIRROR` |
| **Post-flight analysis** | Point it at an SD card or folder. It processes the whole recording as fast as possible and gives a timeline, unique counts and clickable best frames. | `REPLAY ANALYSIS` |
| **Simulate** | Public-domain/CC0 sample clips, so it works with no hardware and no internet. | `SIMULATED` |

The badge is always drawn on the video, and the header shows the same thing. Recorded footage
is never presented as live.

> **Scope:** observation, counting and reporting only. Nothing here
> controls, arms or steers a drone.
>
> **Privacy:** scene description is **offline by default**. It uses a local VLM on the HQ Mac,
> or the detector-only summary. If you set `GEMINI_API_KEY`, Gemini becomes a fallback and
> **snapshot JPEGs are sent to Google**, so use demo footage only.

---

## Contents

1. [Office demo quick start (Drone-ops Mac)](#office-demo-quick-start-drone-ops-mac)
2. [Running on a laptop](#running-on-a-laptop)
3. [Replay as live](#replay-as-live)
4. [Live inputs: what DJI goggles can and can't do](#live-inputs-what-dji-goggles-can-and-cant-do)
5. [Post-flight analysis](#post-flight-analysis)
6. [Scene description: local VLM, Gemini, offline](#scene-description-local-vlm-gemini-offline)
7. [Performance: measured](#performance-measured)
8. [Spot reports: ATAK and mesh](#spot-reports-atak-and-mesh)
9. [Heartbeat](#heartbeat)
10. [Why detections are not geolocated](#why-detections-are-not-geolocated)
11. [Demo script](#demo-script)
12. [Raspberry Pi install](#raspberry-pi-install)
13. [Detection, tracking and counting](#detection-tracking-and-counting)
14. [Configuration reference](#configuration-reference)
15. [API](#api)
16. [Development](#development)

---

## Office demo quick start (Drone-ops Mac)

```bash
git clone https://github.com/Jacob-Haynes/fpv-vision ~/Repos/isr-kit/fpv-vision
cd ~/Repos/isr-kit/fpv-vision
mkdir -p ~/Movies/FPV && cp /Volumes/<card>/DCIM/*/*.MP4 ~/Movies/FPV/   # your O3/O4 footage
scripts/run-laptop.sh --setup-only        # one-off: venv + torch/ultralytics (needs internet)
scripts/run-laptop.sh                      # http://isr-droneops.local:8110
```

The first run downloads the YOLO26s weights (19 MB) and the sample clips into
`~/Library/Application Support/isr-kit/fpv-vision`. After that **no internet is needed**.
If `~/Movies/FPV` contains footage, the app starts replaying it, looped. Otherwise it plays the
simulate clip.

## Running on a laptop

`scripts/run-laptop.sh` (macOS or Linux) does the following:

- Creates `.venv` with `uv` if it is installed, otherwise with Python 3.10–3.13.
- Installs `requirements-desktop.txt`, and reinstalls only when the requirements change.
- Runs **in the foreground** until SIGTERM or Ctrl-C, then exits cleanly in about 3 s and
  publishes heartbeat `stopped`.
- Binds `0.0.0.0:8110`.

It reads configuration **only from the environment**. The platform's `laptop/start-mac.sh`
sources `~/.config/isr-kit/station.env` first (see `deploy/station.env.example`). Defaults it
applies when a variable is unset:

| variable | Mac default |
|---|---|
| `TAK_HOST` / `TAK_PORT` | `isr-pi.local` / 8087 |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_USER` | `isr-pi.local` / 1883 / `isrkit` |
| `MQTT_PASS` | (from env; a warning is printed if missing) |
| `VLM_URL` / `VLM_MODEL` | `http://isr-hq.local:11434` / `qwen3-vl:8b-instruct` |
| `FPV_DEVICE` | `auto`, which picks **CUDA > Apple MPS > CPU** |
| `FPV_MODEL` | `auto`: **YOLO26s @ 640 on a GPU**, YOLO26n @ 416 on CPU |
| `FPV_FOOTAGE_DIRS` | `~/Movies/FPV` (folders listed in the UI's footage picker) |
| `FPV_DATA_DIR` | `~/Library/Application Support/isr-kit/fpv-vision` |
| `FPV_TARGET_FPS` / `FPV_STREAM_WIDTH` | 30 / 1280 |

On the Pi, local tools use `guest`/`guest`. RabbitMQ's `guest` account only works over loopback,
so the Macs must use `isrkit`, which pi-platform creates.

**macOS permissions** (one-off, per terminal app):

- **Camera** (for the built-in camera, iPhone Continuity Camera or a USB capture card).
  **Start fpv-vision from Terminal.app** (or iTerm). macOS grants camera access to the *app
  that launched the process*. Processes started from inside the Claude app, or from other
  sandboxed or helper launchers, are **denied silently**: the camera opens but delivers
  nothing, and no prompt appears. Running from Terminal.app is confirmed to work on the
  Drone-ops Mac. fpv-vision asks for access at startup; if you miss the prompt, go to *System
  Settings → Privacy & Security → Camera*, allow Terminal, and restart. The UI shows the current
  permission state next to the source.
- **Screen Recording** (only for the screen-mirror source). Go to *System Settings → Privacy &
  Security → Screen & System Audio Recording*, allow your terminal app, and restart it. Without
  it macOS returns black frames, and the UI says so.

Windows/CUDA is not a priority for this demo. The code picks CUDA automatically on an NVIDIA
machine. On Windows, install a CUDA build of torch from pytorch.org first and run
`python -m fpv_vision` from a venv; there is no `.ps1` launcher. On Windows, cameras are opened
by index with DirectShow.

## Replay as live

Pick footage in the **Footage** panel (tick one or more files, then press **▶ Replay ticked**),
or start with `--replay ~/Movies/FPV/a.mp4 ~/Movies/FPV/b.mp4`, or set
`FPV_SOURCE=replay:/path/a.mp4|/path/folder`.

- **Real time, never slow motion.** Each file is timed to the wall clock at its native fps. If
  decode or inference falls behind, frames are skipped (`grab()`, and a seek if more than 2 s
  behind), so it behaves like a live feed. Detection runs on the newest frame. The UI shows
  "N frames dropped to stay real-time".
- **Playlist** loops forever. ⏮ / ⏭ move between files, ⏸ pauses, and the **scrub bar** seeks.
- **Full-res decode, downscaled inference.** 4K frames are shrunk to 1280 px wide before YOLO
  (`FPV_INFER_MAX_WIDTH`), and boxes are mapped back. The stream is shown at
  `FPV_STREAM_WIDTH` (1280). Snapshots keep the full-resolution raw frame.
- Detection, tracking, counts, scene description, recording and spot reports all work exactly
  as on a live feed. Spot reports from replayed footage say so in the CoT remarks
  ("From recorded footage: file t=…").
- **Use the air unit's recording, not the goggles' DVR.** N3 DVR files have the OSD burned in by
  default, and OSD text causes false detections. Use the crop tool if you have to.

## Live inputs: what DJI goggles can and can't do

Researched from DJI spec pages, FAQs and manuals (2026-11). V means stated by DJI; the rest is
inference or third-party.

| Goggles | Link | HDMI out | USB-C video/UVC | Live view to phone (DJI Fly) | RTMP via DJI Fly |
|---|---|---|---|---|---|
| **Goggles N3** (user's) | O4 only | **No** (V) | **No** (V, not documented) | **Yes, USB-C data cable or MFi Lightning** (V) | Probably; not stated for N3 |
| Goggles 3 | O3 + O4 | No | No | Yes, cable or Wi-Fi (V) | Yes (V) |
| Goggles 2 / Integra | O3 (+O4 with FPV RC 2) | No | No | Yes, cable | Yes, DJI Fly (3P) |

All of them can relay HDMI through a DJI RC Pro (V). Unofficial firmware (fpv.wtf/WTFOS, USB
video hacks) is deliberately not used.

**So with the N3 there is no direct live feed into a computer.** The honest options, from best to worst:

1. **Replay as live** with real footage. This is the headline: reliable, and it runs in real time.
2. **Phone mirroring + screen capture (best effort, added latency).** The goggles go over USB-C
   to the phone, and you open DJI Fly → GO FLY. Then mirror the phone to the Mac:
   - **iPhone:** use AirPlay *Screen Mirroring* to the Mac (Control Centre → Screen Mirroring →
     the Mac; the Mac needs AirPlay Receiver on). Or connect the iPhone by cable and use
     QuickTime → *File → New Movie Recording* → camera = the iPhone, full screen.
   - **Android:** `scrcpy`.

   In fpv-vision choose **Screen capture** and press **✂ Crop**, then drag around the video
   area so the DJI Fly UI is excluded. Detection then runs on the cropped region. Expect extra
   latency from the goggles→phone link plus mirroring, and some mirroring compression. DJI
   doesn't publish these figures, so measure them on the day. This path needs the **Screen
   Recording** permission.
3. **RTMP from DJI Fly** (Transmission → Live Streaming Platforms → RTMP) to an RTMP server on
   the Mac (e.g. `mediamtx`), then use source `rtmp://localhost/live/drone`. This adds several
   seconds of latency and is **not confirmed for the N3**. Test it before relying on it.
4. **Live real-time boxes without the drone:** the **GoPro HERO4 → HDMI capture card** (below),
   the Mac's camera, or an **iPhone through Continuity Camera**. These appear by name in the
   source list ("Camera: iPhone…", "Camera: <capture card name>"), so no index guessing.

### GoPro HERO4 → Guermok HDMI capture card → Mac (live)

```
 GoPro HERO4 ──micro-HDMI (Type D) → HDMI cable──> Guermok HDMI→USB-C capture card (UVC, USB 3.0)
             ──USB-C, straight into a Mac USB-C/Thunderbolt port (NOT a hub)──> Drone-ops Mac
```

- **Cable:** the HERO4's HDMI port is **micro-HDMI**. You need a **micro-HDMI (Type D) to
  full-size HDMI** cable. GoPro doesn't include one ("micro HDMI, composite or combo cable, sold
  separately" in the [HERO4 Black manual](https://gopro.com/content/dam/help/hero4-black/music-bundle-manuals/UM_H4Black-Music_ENG_REVB_WEB.pdf)).
  Some cases cover the port, so use the frame or an open side door. The HERO4 has **no USB
  webcam mode**. The GoPro Wi-Fi preview is deliberately not supported, because joining the
  camera's Wi-Fi would take the Mac off the Pi hotspot.
- **Camera settings:**
  - **Setup → On-Screen Display → Off.** The manual describes OSD as the recording icons and
    file info "on the viewing screen during playback". Users report it also governs the
    overlays on the live HDMI feed. Check before the demo, because overlay text causes false
    detections; the **✂ Crop** tool is the fallback.
  - **Setup → Video Format → PAL** in the UK (25/50 fps; NTSC is 30/60). The manual says this
    governs the TV/HDMI frame rates.
  - Put the camera in **Video mode, not recording**. Keep it on external power: the battery
    drains fast with HDMI out and the screen on.
- **Measured on the Drone-ops Mac:** HERO4 set to 1080p30 → micro-HDMI → Guermok "USB3 Video"
  (AVFoundation index 1, so `camera:USB3 Video`). It gives a clean feed with no OSD in frame.
  **The card always outputs 1920×1080 @ 60 fps, but only 30 of those are real**; every second
  frame is a copy. fpv-vision **drops padded duplicates** before inference. It compares each
  frame with the previous one on a strided full-resolution sample (`FPV_DEDUP_THRESHOLD`,
  default 0.5 mean absolute difference; `FPV_DEDUP=0` disables it). The UI then shows
  **"receiving 60 / unique 30 fps"**, and detection, tracking and the FPS tiles use only the
  real frames. To calibrate on new hardware, run this from Terminal.app with something moving
  in front of the camera:

  ```bash
  .venv/bin/python -m fpv_vision --probe-camera "camera:USB3 Video"
  ```

  Duplicates cluster near 0; real frames sit well above.
- **Recording resolution ≠ HDMI preview.** The HERO4's video mode (e.g. 4K30, 2.7K, 1080p60)
  sets what it **records to the SD card**. The HDMI port sends the **live preview**, at a
  resolution and frame rate the camera chooses, and the capture card then scales or pads that
  to its own output mode. So 1080p30 on the camera became "1080p60 with duplicates" on the
  card. Trust the UI's readout, not the camera menu.
- **What resolution arrives (other setups):** GoPro doesn't publish the HERO4's live-preview HDMI resolution.
  The manual only says HDMI is certified up to 1080p. One forum user saw 720×480 from a HERO4 on
  their capture setup, and others report 1080p. **The UI shows what the card actually
  negotiated** (source pill and "device mode …, receiving N fps"). If it reads 720×480, set the
  camera to a 1080p video mode, unplug and replug the HDMI cable, and check again.
- **Capture defaults:** fpv-vision asks cameras and capture cards for **1920×1080 at 60 fps**
  (`FPV_CAPTURE_WIDTH/HEIGHT/FPS`). On Linux and Windows it requests **MJPEG**, which needs
  less USB bandwidth and CPU than raw YUY2. On macOS, AVFoundation picks the device format
  itself (and decodes MJPEG in hardware), so the format isn't forced there. Plug the card
  **directly into a Mac USB-C port**, not a hub or dock, so it gets USB 3 bandwidth for
  1080p60.
- **Recorded HERO4 MP4s** (H.264, no GPS) work in **replay as live** and **post-flight
  analysis** like any other MP4. Copy them from `DCIM/100GOPRO/` into `~/Movies/FPV`.

## Post-flight analysis

In the UI's **Post-flight analysis** panel, enter a path (the mounted SD card folder, e.g.
`/Volumes/Untitled/DCIM/DJI_001`) or tick files, choose a sample rate, and press **Analyse**.

- It runs in the background with its own detector, so the replay or live view keeps running.
- Output:
  - a per-second **timeline** of people and vehicles; click it to replay from that moment,
  - **unique tracked counts**,
  - up to 12 **best frames**, at least 3 s apart. Best frames are saved as normal snapshots, so
    **describe** and **send spot report** work on them.
- Results are also written to `<data>/analysis/*.json`.
- Speed on the M4 Max test machine: 30 s of **1080p60 in 6 s (5×)** and **4K60 in 11 s (2.7×)**.
  This is decode-bound, so the sample rate barely matters for 4K. A 5-minute 4K flight takes
  about 2 minutes. The M4 Pro will be somewhat slower.

## Scene description: local VLM, Gemini, offline

`VLM_BACKEND` = `auto` (default) | `openai_compat` | `gemini` | `off`.

- `openai_compat` uses an **Ollama** server on the HQ Mac. `VLM_URL` is the **bare** base URL
  (`http://isr-hq.local:11434`), and we append `/v1/chat/completions`. The image is sent as a
  base64 `image_url`. This works with **no internet**.
- If the selected backend is unreachable, the fallback order is: **openai_compat → gemini (only
  if `GEMINI_API_KEY` is set) → off** (local detector summary such as "2 people near a car, left
  of frame"). The header pill shows the active backend, and each summary is labelled with what
  produced it.
- `auto` picks `openai_compat` when `VLM_URL` is set, else `gemini` with a key, else `off`.
- Rate limit: `VLM_MIN_INTERVAL` (3 s) for the local VLM, `GEMINI_MIN_INTERVAL` (10 s) for
  Gemini. An optional auto-summary every N seconds is set in the UI; it skips empty scenes.

**HQ Mac (M5 Pro, 64 GB) setup.** The default model is
[`qwen3-vl:8b-instruct`](https://ollama.com/library/qwen3-vl/tags): 6.2 GB, verified on ollama.com, and it
needs Ollama ≥ 0.12.7.

```bash
ollama pull qwen3-vl:8b-instruct
# Ollama only listens on localhost by default. Make it reachable on the hotspot:
launchctl setenv OLLAMA_HOST 0.0.0.0:11434   # then restart the Ollama app
curl http://isr-hq.local:11434/v1/models     # from the Drone-ops Mac
```

Larger vision models that still fit comfortably in 64 GB, if you want richer descriptions and
can accept slower replies:

| model | size |
|---|---|
| `qwen3-vl:30b` | 20 GB |
| `qwen2.5vl:32b` | 21 GB |
| `gemma3:27b` | 17 GB |

Set `VLM_MODEL` to use one. Use an **instruct** build: the stock `qwen3-vl:8b` tag is a
"thinking" model. It thinks even with `think:false`, takes far longer and sometimes returns an
empty answer, whereas `qwen3-vl:8b-instruct` answers in 1–5 s. An empty reply counts as a
failure, so the chain falls back to Gemini or the local summary. Replies are capped at 300
tokens, and we never request Ollama's `format: "json"`, because MLX-runner models reject it
with HTTP 501.

## Performance: measured

The test machine was an **Apple M4 Max** (16-core CPU, 40-core GPU), with no other heavy load. It
ran the full app: decode → detect + ByteTrack → annotate → 1280 px MJPEG. The test files were
30 s 60 fps clips made to look like DJI files: H.264 at about 120 Mb/s, HEVC at about 100 Mb/s,
with high detail and noise. Status figures were read after about 15 s.

| footage | model (MPS, 640) | source decode | detect fps | inference | latency* | dropped |
|---|---|---|---|---|---|---|
| 4K60 HEVC | yolo26n | 60.1 fps | 59.8 | 17 ms | 43 ms | 33 (startup) |
| 4K60 HEVC | **yolo26s** (default) | 60.0 | 58.0 | 18 ms | 39 ms | 33 |
| 4K60 HEVC | yolo26m | 60.1 | 54.2 | 20 ms | 42 ms | 36 |
| 4K60 H.264 | yolo26s | 60.0 | 60.0 | 16 ms | 38 ms | 45 |
| 1080p60 H.264 | yolo26s | 60.0 | 42.6 | 23 ms | 59 ms | 11 |
| 1080p60 H.264 | yolo26n | 60.1 | 46.2 | 20 ms | 55 ms | 10 |

\*Latency is the time from frame decoded to annotated JPEG ready, plus the age of the boxes drawn
on it. It excludes the capture device, the network and the browser. The UI shows these values
live.

Other results from the same machine:

- The video stream held about 27–30 fps against its 30 fps cap. CPU use was about 3.5 cores
  for 4K60 (mostly software HEVC/H.264 decoding).
- Raw OpenCV decode capacity: 4K60 about 110–120 fps, 1080p60 about 350 fps.
- Detector-only microbenchmark at 640 on 1080p frames: on MPS, n 21 ms, s 20 ms, m 27 ms; on CPU
  (PyTorch), n 35 ms, s 56 ms.

**M4 Pro (Drone-ops Mac): estimated, not yet measured.** It has half the GPU cores and fewer CPU
cores. I expect about 30–45 detect fps with yolo26s and real-time 4K60 decode with some headroom.
**YOLO26s is the default because it keeps comfortably above 25 detect fps there.** Use yolo26m
(`FPV_MODEL=yolo26m`) if `--benchmark` shows headroom. To measure on the machine:

```bash
scripts/run-laptop.sh --replay ~/Movies/FPV/<4k clip>.MP4 --benchmark 200
```

The UI's tiles show the live figures during the demo: video fps, detect fps, ms per inference,
latency and frames per detection.

**Raspberry Pi 5 (CPU, NCNN).** This is an estimate, not measured. Ultralytics' Pi 5 benchmark
puts YOLO26n NCNN at about 67 ms at 640 px
([guide](https://docs.ultralytics.com/guides/raspberry-pi)). At 416 px expect about 10–20
detections/s with a 15–20 fps overlay. 4K replay is not realistic on the Pi; use 1080p or less.

## Spot reports: ATAK and mesh

The **Send spot report** button builds a SALUTE report from the selected snapshot, or from a
fresh one: Size and Equipment come from the detections, Activity is free text plus the summary,
Location is **entered by the operator** (MGRS or lat/lon). If the operator leaves it empty it
falls back to `FPV_DEFAULT_POSITION`, then to the shared `SENSOR_LAT`/`SENSOR_LON`. With none of
those set the report is refused: **we never emit 0,0**. Time is a DTG. It then sends:

1. **CoT** to `TAK_HOST:TAK_PORT`: an `a-u-G` marker with `how="h-e"` (human-entered) and the full
   report in its remarks. It goes over fpv-vision's one long-lived, registered TAK connection
   (see below). Reports are queued until registration completes, and while TAK is down.
2. A **mesh text** of 200 characters or less on MQTT `meshbus/tx`, as `{"text","to":null,"channel":0,"source":"fpv-vision"}`.
   The blufor-tracker bridge owns the radio, and we never open its serial port. Replies on
   `meshbus/rx` show in the **Mesh inbox**.

**TAK connection (contract C5).** OpenTAKServer 1.8.3 drops CoT from anything that isn't
a registered EUD. It only binds a client whose first `<contact>` arrives after its channel
opens (about 3 s after connect), and it silently refuses a uid whose callsign is already
taken. So `fpv_vision/tak_client.py` does the following:
- opens **one** TCP connection at startup and reconnects with backoff;
- **3 s after every (re)connect**, sends a registration SA, then re-sends it every 60 s.
  The SA's uid is `isrkit-fpv-vision-<ISR_HOST>`, its callsign `FPV <ISR_HOST>` (stable,
  never random), type `a-f-G-E-S`, with `endpoint="*:-1:stcp"`, group Cyan / Team Member,
  and `takv platform="isr-kit"`. Its position is `SENSOR_LAT`/`SENSOR_LON` with stale
  now+120 s. With no position set it uses `0,0` with unknown hae/ce/le and stale = start,
  so ATAK never draws it. This is the only 0,0 the kit emits;
- sends spot reports on that same connection, and only after registration;
- reads and discards what OTS sends back, so the server never blocks on us.

Spot-report markers are also built so OTS **stores** them, which puts them on OTS's web map
(contract C5.5):
- type `a-u-G`;
- no `<takv>` and no `<contact endpoint>`;
- never at 0,0;
- `<link>` only to our own registration uid, with `parent_callsign` and `precisionlocation`
  set to USER for the operator-entered position.

The header pill reads "TAK registered as FPV <host>" once the SA has gone out. The tests in
`tests/test_tak_client.py` run against a fake OTS (`tests/fake_ots.py`) that enforces all
three rules.

**Provenance labels (contract C4.3).** Reports made from the sample clips start with
`SIMULATED: ` in the CoT remarks and `[SIM] ` in the mesh text. Reports from replayed or
post-flight footage start with `RECORDED: ` and `[REC] `. Reports from live capture (camera,
capture card, screen mirror, network stream) have no prefix. The label is stored with each
snapshot when it is taken, so a recorded best frame stays `[REC]` even after you switch to a
live source.

**Rate limits (contract C4.4).**
- At most **1 mesh text per 30 s**. A second spot report inside that window still sends its
  CoT, and the UI says "mesh rate limit … retry in N s".
- At most **1 CoT per 5 s per uid**. Every spot report has its own uid, so this only stops
  accidental double-sends.

## Heartbeat

Every 10 s, fpv-vision publishes this retained message to `isrkit/heartbeat/fpv-vision`:

```json
{"tool":"fpv-vision","host":"isr-droneops","url":"http://isr-droneops.local:8110/","mode":"replay","status":"ok","ts":1794836403,"detail":"yolo26s mps 2160p 58fps, VLM qwen3-vl:8b-instruct @ isr-hq"}
```

- `mode` is `replay` | `simulate` | `mps` | `cuda` | `cpu`, which is the video source type, or
  the accelerator when the input is live.
- `detail` is optional and human-readable, e.g. `"yolo26s mps 1080p 58fps, VLM qwen3-vl:8b-instruct @ isr-hq"`.
- `status` is `ok`, `degraded` (no detector, or the source is reconnecting) or `error`.
- On a clean exit it publishes `stopped`. The MQTT last-will publishes `error` if the process dies.
- `host` follows contract C4.1: env `ISR_HOST` if set, otherwise the lowercased short
  `socket.gethostname()`. The `url` is `http://<host>.local:<port>/`; override it with
  `FPV_PUBLIC_URL`.

## Why detections are not geolocated

The CoT marker goes at the **position the operator typed**, never at a computed location for
the boxed objects. Turning a pixel into a ground coordinate needs all of these:

- the drone's GPS position and height above ground,
- the camera's pose (gimbal or tilt plus aircraft attitude), time-synced to each frame,
- lens intrinsics,
- a terrain model.

O3/O4 air units record no GPS (their `.SRT` sidecars carry link data, per third-party reports).
A fixed-tilt FPV camera's pose changes with throttle. Faking it would put confident markers in
the wrong place.

## Demo script

About 10 minutes, in the office. The Drone-ops Mac runs fpv-vision; the audience watches it on
the big screen or on a tablet at `http://isr-droneops.local:8110`.

1. **Frame it (1 min).** Point at the header and the badge in the top-right of the video:
   **REPLAY** means real footage from our own flights, replayed in real time. Then point at the
   detector pill (`yolo26s · pytorch/mps`): "on the laptop's GPU, no cloud".
2. **Replay as live (3 min).** Play a playlist of 2–3 flights. Show boxes, track IDs and the
   *unique* count compared with *in view*. Point at the tiles: 60 fps source, detect fps, about
   40 ms latency. Scrub to an interesting moment and pause on it.
3. **Describe (1 min).** Press **Describe scene**. The pill shows `scene: VLM qwen3-vl:8b-instruct`
   (running on the HQ Mac, offline). Then quit Ollama on the HQ Mac to show it fall back to the local
   summary.
4. **Report (2 min).** Select the snapshot, enter the observer's grid, and press **Send
   spot report**. The marker appears in ATAK, and the SPOTREP text arrives on a Heltec node.
5. **Post-flight (2 min).** Pretend we just landed: point **Analyse** at the SD card folder.
   About 2 minutes later you have a timeline and best frames. Click a peak to jump the replay
   there, and spot-report a best frame.
6. **Live (1 min).** Switch the source to the **capture card** (GoPro HERO4 on HDMI) or **Camera:
   iPhone** (Continuity Camera), and walk through the room for live boxes. If time allows, show the N3 → phone → screen-mirror
   path and be clear that it is best-effort.
7. **Discuss.** Why the marker is at the *entered* grid. What the cloud fallback would send off
   the device. What it would take to geolocate detections.

Reset between groups with **Reset counts**.

**Failure drills:**

- No TAK or MQTT: the pills go amber and spot reports queue or report "not sent". Video carries on.
- No HQ Mac: scene description falls back to the local summary.
- No footage: it falls back to the sample clip.

## Raspberry Pi install

The Pi path is unchanged: YOLO26n NCNN on the CPU, as a pi-profiles unit.

```bash
git clone https://github.com/Jacob-Haynes/fpv-vision && cd fpv-vision
sudo bash deploy/install.sh        # idempotent, non-interactive, does NOT enable at boot
sudo nano /etc/fpv-vision.env      # FPV_SOURCE, FPV_DEFAULT_POSITION, optional VLM_URL
sudo pi-profiles fpv-vision        # combined profiles (e.g. with OpenTAKServer + mesh bridge) live in pi-profiles
```

`install.sh` does the following:

- Installs `python3-venv`, `v4l-utils`, `ffmpeg` and `libgl1`.
- Creates the `fpv-vision` user (in the `video` group).
- Syncs the code to `/opt/fpv-vision` (skipped when run from there) and builds a venv.
- Creates `/var/lib/fpv-vision`, and creates `/etc/fpv-vision.env` (mode 600) only if it doesn't
  exist.
- Exports YOLO26n to NCNN, fetches the sample clips, and installs the unit
  (`TimeoutStopSec=15`) and `profiles.d/*.conf`.

Profiles are tagged `# @devices camera`.

Pi sources:

| spec | source |
|---|---|
| `uvc:/dev/video0` | USB capture card (analog FPV receiver AV-out, an HDMI source such as the GoPro HERO4's micro-HDMI) |
| `rtsp://…` / `rtmp://…` | network streams |
| `replay:/path` | footage files |

`FPV_SOURCE=auto` uses the first capture card it finds, or the sample clip if there is none.

```
 analog FPV goggles/receiver ──AV out──> AV→USB dongle (UVC) ──USB──> Pi 5 / Mac
 HDMI source (GoPro HERO4, DJI RC Pro) ──HDMI──> HDMI→USB capture card (UVC, MJPEG) ──USB 3──> Pi 5 / Mac
```

## Detection, tracking and counting

- **Models:** Ultralytics YOLO26 (s on GPU, n on CPU). Override with `FPV_MODEL`, `FPV_GPU_MODEL`
  or `FPV_CPU_MODEL`. On CPU it prefers an NCNN export, then ONNX, then `.pt`
  (`python -m fpv_vision.export`). On GPU it uses `.pt`. Weights download once into
  `<data>/models`.
- **Classes:** people, vehicles (bicycle, car, motorcycle, bus, train, truck), boats and aircraft.
  Set `FPV_ALL_CLASSES=1` for all 80 COCO classes. People are drawn red, vehicles orange.
- **Adaptive frame-skip:** the detector always takes the newest frame. The overlay is drawn on
  every frame with the latest boxes. If the overlay drops below 85% of its target, the detector
  backs off.
- **Counting:** "unique" counts ByteTrack IDs seen at least 3 times. There is no
  re-identification, so a person who leaves and comes back counts again, and a looping replay
  counts again every loop. Use **Reset counts**.
- **Limits worth teaching:**
  - small objects seen from altitude;
  - OSD overlays;
  - motion blur;
  - COCO classes, which are everyday objects, not military vehicles or camouflage.

## Configuration reference

Configuration comes from the environment (`/etc/fpv-vision.env` on the Pi, or `station.env` /
shell on a Mac). Matching CLI flags override it: `--simulate`, `--replay PATH…`, `--source`,
`--port`, `--host`, `--model`, `--imgsz`, `--device`, `--no-detect`, `--data-dir`, `--tak-host`,
`--tak-port`, `--mqtt-host`, `--mqtt-port`, `--mqtt-user`, `--mqtt-pass`, `--vlm-url`,
`--vlm-backend`, `--no-tak`, `--no-mesh`, `--benchmark N` and `--list-cameras`.

| variable | default | meaning |
|---|---|---|
| `FPV_CAPTURE_WIDTH` / `FPV_CAPTURE_HEIGHT` / `FPV_CAPTURE_FPS` | 1920 / 1080 / 60 | requested camera/capture mode (Pi env: 1280×720@30) |
| `FPV_SOURCE` | `simulate` | `replay:…`, `camera:<name>`, `uvc:N`, `screen:N`, `rtsp://…`, `simulate[:aerial]`, `auto` |
| `FPV_FOOTAGE_DIRS` | `~/Movies/FPV` | folders shown in the footage picker (`:`-separated) |
| `FPV_DEVICE` | `auto` | CUDA > MPS > CPU |
| `FPV_MODEL` | `auto` | or a model name or path |
| `FPV_GPU_MODEL` / `FPV_GPU_IMGSZ` | `yolo26s` / 640 | used when `FPV_MODEL=auto` |
| `FPV_CPU_MODEL` / `FPV_CPU_IMGSZ` | `yolo26n` / 416 | used when `FPV_MODEL=auto` |
| `FPV_IMGSZ` | 0 | 0 = auto |
| `FPV_INFER_MAX_WIDTH` | 1280 | downscale big frames before inference |
| `FPV_CONF` / `FPV_ALL_CLASSES` | 0.30 / 0 | |
| `FPV_TARGET_FPS` / `FPV_STREAM_WIDTH` / `FPV_JPEG_QUALITY` | 20 / 960 / 75 | (30 / 1280 via run-laptop.sh) |
| `FPV_ALLOW_SCREEN` / `FPV_REQUEST_CAMERA` | 1 on macOS / 1 | screen source; macOS camera prompt |
| `FPV_DATA_DIR` | `/var/lib/fpv-vision` or `./data` | snapshots, clips, models, analysis |
| `FPV_CALLSIGN` / `FPV_DEFAULT_POSITION` | `FPV-VISION` / – | spot reports (position falls back to `SENSOR_LAT`/`SENSOR_LON`) |
| `TAK_HOST` / `TAK_PORT` / `TAK_ENABLED` | 127.0.0.1 / 8087 / 1 | CoT output |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_USER` / `MQTT_PASS` | 127.0.0.1 / 1883 / guest / guest | Pi defaults (Macs: `isr-pi.local`, `isrkit`) |
| `MQTT_ENABLED` / `MESH_ENABLED` | 1 / 1 | broker connection and heartbeat / mesh sends |
| `ISR_HOST` / `FPV_PUBLIC_URL` | gethostname / `http://<host>.local:<port>/` | heartbeat identity |
| `SENSOR_LAT` / `SENSOR_LON` | – | shared fallback spot-report position (never 0,0) |
| `VLM_BACKEND` / `VLM_URL` / `VLM_MODEL` / `VLM_MIN_INTERVAL` | auto / – / `qwen3-vl:8b-instruct` / 3 | scene description |
| `GEMINI_API_KEY` / `GEMINI_MODEL` / `GEMINI_MIN_INTERVAL` | – / `gemini-3.8-flash` / 10 | cloud fallback |
| `GEMINI_PERIODIC_SECONDS` | 0 | auto-summary period for the active backend |

## API

| method | path | |
|---|---|---|
| GET | `/stream.mjpg`, `/frame.jpg` | annotated stream / latest frame |
| GET | `/api/status` | everything the UI shows (stats, latency, source, replay state, vlm, tak, mesh, heartbeat) |
| GET/POST | `/api/sources`, `/api/cameras`, `/api/footage`, `/api/source {spec}` | sources |
| POST | `/api/replay {action: seek\|pause\|next\|prev, value}` | replay control |
| POST | `/api/crop {crop: [x,y,w,h] \| null}` | crop (normalised 0–1) |
| POST | `/api/snapshot`, `/api/describe {mode}`, `/api/snapshots/<name>/describe` | snapshots and descriptions |
| GET | `/api/snapshots`, `/media/snapshots/<file>` | gallery |
| POST/GET | `/api/analysis {target, sample_fps}`, `/api/analysis/<id>`, `/api/analysis/<id>/cancel` | post-flight analysis |
| POST | `/api/record/start\|stop`; GET `/api/clips`, `/media/clips/<file>` | clips (mp4v) |
| POST | `/api/periodic {seconds}`, `/api/spot-report {…}`, `/api/counts/reset` | |

## Development

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-desktop.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The tests need no hardware, network or weights:

- They use a fake detector, synthetic and generated clips, a fake Ollama HTTP server and a fake
  MQTT client.
- `tests/test_yolo_optional.py` also runs the real model when `data/models` and the samples exist.

Code layout (`fpv_vision/`):

| file | role |
|---|---|
| `sources.py` | replay, cameras, screen capture, network sources and crop |
| `detector.py` | device and model selection, YOLO and ByteTrack |
| `pipeline.py` | detect and render threads, latency figures, recorder, snapshots |
| `analysis.py` | post-flight analysis |
| `vlm.py` + `gemini.py` | scene description chain |
| `mesh.py` | MQTT: mesh texts and the heartbeat |
| `cot.py`, `spotrep.py`, `geo.py` | ATAK output, SALUTE reports, MGRS |
| `service.py` | glue between the parts |
| `web/` | web UI |

## License

Copyright (C) 2026 Jacob Haynes. Licensed under the GNU AGPL v3 (see LICENSE). Uses Ultralytics YOLO (AGPL-3.0); model weights are subject to Ultralytics' licence.

The AGPL means that if you modify fpv-vision and let other people use it over a network
(for example by hosting the web UI), you must offer them the source of your modified
version. The sample clips are not part of this repository. They are downloaded at runtime
from Wikimedia Commons under their own terms (NASA public domain; CC0).
