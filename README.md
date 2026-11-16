# fpv-vision

Live AI object detection on a drone or handheld video feed. It is a drone-ISR training
demo that runs on one Raspberry Pi 5 with **no AI accelerator**: YOLO runs on the CPU.
Google Gemini is optional; when you configure it, it writes plain-language scene
summaries.

- **Video in:** a USB capture card (an analog FPV receiver, goggles HDMI-out, or GoPro HDMI-out), GoPro USB webcam mode, RTSP/RTMP/HTTP/UDP streams, video files, or `--simulate`.
- **Detection:** Ultralytics **YOLO26n** exported to **NCNN**, with **ByteTrack** tracking. People are shown in red, vehicles in orange. Counts are of unique tracks, so one person walking through the frame counts once.
- **Smooth overlay:** the detector always takes the newest frame and skips the rest. The video is annotated at full rate with the latest boxes. If the overlay frame rate drops, the detector backs off.
- **Web UI on :8110** for a laptop or tablet. It shows the annotated MJPEG stream, live and unique counts, a snapshot gallery, clip recording, a scene summary, and spot reports.
- **ATAK and mesh:** **Send spot report** posts a CoT marker at an *operator-entered* grid, with a SALUTE text and the summary in the remarks. It also sends a short SALUTE text over the mesh using `meshbus/tx`.
- **Offline by default.** Every feature works with no internet. The header shows `LOCAL ONLY (offline)` or `LOCAL + GEMINI`.

> **Scope:** this is a defensive ISR training demo for observation and reporting only. It never
> controls, arms or steers a drone.

> **Cloud mode privacy:** when `GEMINI_API_KEY` is set, **Describe scene** and the optional
> auto-summary **upload snapshot JPEGs to Google**. Use only demo footage, never real
> operational imagery.

---

## Contents

1. [Quick start on a Mac (simulate)](#quick-start-on-a-mac-simulate)
2. [Install on the Pi](#install-on-the-pi)
3. [Video sources and wiring](#video-sources-and-wiring)
4. [Performance (FPS)](#performance-fps)
5. [Detection, tracking and counting](#detection-tracking-and-counting)
6. [Gemini mode](#gemini-mode)
7. [Spot reports: ATAK and mesh](#spot-reports-atak-and-mesh)
8. [Why detections are not geolocated](#why-detections-are-not-geolocated)
9. [Demo script](#demo-script)
10. [Configuration reference](#configuration-reference)
11. [API](#api)
12. [Development](#development)

---

## Quick start on a Mac (simulate)

```bash
git clone https://github.com/Jacob-Haynes/fpv-vision && cd fpv-vision
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m fpv_vision.export --data-dir data      # downloads yolo26n, exports to NCNN
.venv/bin/python -m fpv_vision --simulate --data-dir data   # http://localhost:8110
```

The first `--simulate` run downloads a ~9 MB sample clip into `data/samples/`. If there is no
internet, it shows a synthetic test pattern instead, so the UI still works but nothing is
detected. Sample clips (you can switch between them in the UI):

| spec | clip | licence |
|---|---|---|
| `simulate` (default) | Handheld, eye level: the Yonge & Dundas scramble crossing in Toronto. Many people, cars and bikes. [Commons page](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm) | CC0 (Raysonho @ Open Grid Scheduler / Grid Engine) |
| `simulate:aerial` | NASA drone pass over the Banana Creek stands at the Artemis II launch: buses, cars and crowds, from high up. [Commons page](https://commons.wikimedia.org/wiki/File:Artemis_II_Live_Launch_Coverage_-_Drone,_Crowd_Shots_of_Banana_Creek_(KSC-20260401-MH-AJN01-0001-Artemis_II_Launch_ISO_Drone_Crowd_Shot_Banana_Creek_4K).webm) | Public domain (NASA KSC) |

The aerial clip is a useful teaching point. From that height a nano model at 416 px finds
some of the vehicles but almost none of the people, because people are only a few pixels
tall. See [Detection](#detection-tracking-and-counting).

## Install on the Pi

On a Pi 5 running 64-bit Pi OS Bookworm (Python 3.11), with internet for the first install:

```bash
git clone https://github.com/Jacob-Haynes/fpv-vision && cd fpv-vision
sudo bash deploy/install.sh
sudo nano /etc/fpv-vision.env           # FPV_SOURCE, FPV_DEFAULT_POSITION, optional GEMINI_API_KEY
sudo pi-profiles fpv-vision             # or: sudo pi-profiles bft-fpv  (with OpenTAKServer + mesh bridge)
```

`install.sh` is idempotent and non-interactive. It does these things:

- Installs apt packages: `python3-venv`, `v4l-utils`, `ffmpeg`, `libgl1`, `rsync`.
- Creates the `fpv-vision` system user, in the `video` group.
- Copies the code to `/opt/fpv-vision` and builds a venv in `/opt/fpv-vision/.venv`.
- Creates `/var/lib/fpv-vision/{snapshots,clips,models,samples}`.
- Copies `deploy/fpv-vision.env.example` to `/etc/fpv-vision.env` (mode 600) **only if that file doesn't exist yet**.
- Exports the model to NCNN, downloads the sample clips, and installs the systemd unit and the `profiles.d/*.conf` files for pi-profiles.

It **does not enable the service at boot**; pi-profiles starts and stops it.

Pi profiles shipped in `profiles.d/`:

| profile | units |
|---|---|
| `fpv-vision` | `fpv-vision` |
| `bft-fpv` | `opentakserver`, `mesh-bridge`, `fpv-vision` |

Logs: `journalctl -u fpv-vision -f`. To measure detection speed on the Pi:

```bash
sudo -u fpv-vision /opt/fpv-vision/.venv/bin/python -m fpv_vision --simulate --benchmark 100 --data-dir /var/lib/fpv-vision
```

Cooling: sustained inference keeps all four cores busy. Use the active cooler, or the Pi
5 throttles at about 85 °C and FPS drops.

## Video sources and wiring

Set `FPV_SOURCE` in `/etc/fpv-vision.env`, or pick or type a source in the web UI.
`FPV_SOURCE=auto` uses the first USB capture device it finds, and falls back to the sample clip.

| spec | source |
|---|---|
| `uvc:/dev/video0` (or `uvc:0`) | USB UVC capture device |
| `rtsp://host:8554/stream`, `rtmp://…`, `http(s)://…`, `udp://@:5600`, `srt://…` | network stream, opened with FFmpeg; RTSP uses TCP |
| `file:/path/clip.mp4` | video file, looped at its native rate |
| `gopro-usb` or `gopro-usb:172.2X.1YZ.51` | GoPro in USB webcam mode |
| `simulate`, `simulate:aerial` | sample clips |

Live sources reconnect with backoff. The header pill shows the source status and any error.
To list capture devices on the Pi: `v4l2-ctl --list-devices`. A capture dongle usually shows two
nodes, and the first one is the video. The Pi 5's own `pispbe`/`hevc` nodes are not cameras.

### 1. Analog FPV (5.8 GHz) → USB

```
  5" FPV quad                    ground station                        Raspberry Pi 5
 ┌───────────┐  5.8 GHz analog  ┌──────────────────┐ composite/AV  ┌───────────────┐  USB  ┌───────────┐
 │ FPV camera├──── VTX ~~~~~~~~>│ goggles / 5.8 GHz├──────────────>│ AV→USB dongle ├──────>│ Pi 5      │
 └───────────┘                  │ receiver (AV out)│  (RCA/3.5 mm) │ (UVC, e.g.    │ USB3  │ uvc:/dev/ │
                                └──────────────────┘               │ MS2109-based) │       │  video0   │
                                                                   └───────────────┘       └─────┬─────┘
                                                                         Wi-Fi/Ethernet ─────────┘──> laptop/tablet :8110
```

- Most analog goggles and box receivers have an **AV-out** jack (3.5 mm TRRS or RCA). Feed it into a
  cheap composite-to-USB UVC dongle. Some "OTG" 5.8 GHz receivers (sold for phones) are
  UVC devices themselves, so you can plug them straight into the Pi.
- Analog video is 720×480 or 720×576 interlaced. Set `FPV_CAPTURE_WIDTH=720` and `FPV_CAPTURE_HEIGHT=576`
  (PAL) or `480` (NTSC). Expect noise and breakup at range. That is a good discussion point:
  detection confidence drops in poor video.

### 2. Digital FPV goggles or anything with HDMI out → USB capture card

```
 goggles / monitor / camera ──HDMI──> HDMI→USB capture card (UVC, MS2109 / MS2130) ──USB3──> Pi 5 (uvc:/dev/video0)
```

- Check that your goggles actually have HDMI-out. Not every digital system does, and some
  only output HDMI in certain modes.
- MS2109 cards are USB 2 and give 1080p30 as MJPEG. That is plenty, because the default
  capture is 1280×720 MJPEG. MS2130 cards are USB 3.
- Turn off on-screen display (OSD) overlays on the HDMI feed if you can. OSD text and the
  artificial horizon cause false detections.

### 3. GoPro

**HDMI-out to a capture card is the reliable path.**

```
 GoPro ──micro-HDMI (or Media Mod HDMI)──> HDMI→USB capture card ──USB──> Pi 5 (uvc:/dev/video0)
```

- HERO8 and later need the **Media Mod** for HDMI out. Some older Black models have a
  micro-HDMI port. In the camera settings, set HDMI output to the live/clean feed, not
  playback with overlays.
- Power the GoPro externally for long sessions.

**USB webcam mode (`gopro-usb`) is implemented but not tested on hardware.** On HERO9 and
later, webcam mode makes the camera appear as a **USB network interface**. The Pi gets a
`172.2X.1YZ.5x` address by DHCP, and the camera is `.51` (XYZ are the last three digits of
the serial). fpv-vision looks for that interface, then calls the Open GoPro HTTP endpoint
`http://<cam>:8080/gopro/webcam/start` (it falls back to the HERO8-era `/gp/gpWebcam/START`).
It then reads the MPEG-TS stream the camera sends to UDP port 8554. GoPro's official webcam
app is Mac/Windows only. This path uses the same open HTTP API as the community Linux
scripts, so it depends on firmware. If it fails, use HDMI.

```bash
ip -4 addr | grep 172.2          # should show the GoPro USB interface
curl http://172.2X.1YZ.51:8080/gopro/webcam/status
```

### 4. Network streams (phones, IP cameras, DJI/RTMP, and so on)

```
 IP camera / phone app / NVR ──RTSP──> (Wi-Fi/Ethernet) ──> Pi 5      rtsp://192.168.1.50:554/stream1
 drone app "live stream to RTMP" ──> RTMP server on LAN ──> Pi 5      rtmp://<server>/live/drone
```

RTSP is the simplest option. To receive RTMP *pushed* from a drone app, run a small RTMP
server (for example mediamtx) on the LAN or the Pi, and point fpv-vision at it.

### 5. 360 camera

You can use a 360 camera through its webcam/UVC mode or an RTSP live stream, if it has one.
YOLO runs on whatever frame it gets. On an equirectangular (unwrapped) frame, objects near
the edges and poles are distorted, and detection there is noticeably worse. A single-lens
"flat" view works better.

### 6. Files

To replay recorded flights, pick `file:/var/lib/fpv-vision/clips/<clip>.mp4` or any MP4/WebM.

## Performance (FPS)

**Measured on the Mac** (Apple M4 Max, 16 cores), `python -m fpv_vision --simulate --benchmark 80`.
Each figure is per-frame latency for detection plus ByteTrack, on 854×480 frames at imgsz 416:

| model / backend | ms per frame | FPS |
|---|---|---|
| yolo26n · NCNN | 5.5 | 183 |
| yolo26n · ONNX Runtime | 5.8 | 173 |
| yolo26n · PyTorch | 9.8 | 102 |
| yolo11n · NCNN | 5.7 | 177 |
| yolo11n · ONNX Runtime | 6.3 | 158 |

With the full app running on the Mac (capture, detect, annotate, JPEG, MJPEG), the overlay
held the 20 FPS cap (about 18.5 measured), and the detector ran on every frame from the
24 FPS clip.

**Pi 5 estimate. This is not measured: the author had no Pi on hand when writing this.**
Ultralytics' own Pi 5 benchmarks put YOLO26n NCNN at about **67 ms per image at 640 px**
([Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi)). Users have reported
up to about 150 ms on less tuned setups. Scaling by pixel count to 416 px (×0.42), and
adding tracking and pre-/post-processing:

| setting | detector | overlay / stream |
|---|---|---|
| imgsz 416 (default), NCNN | **≈ 10–20 detections/s** (50–100 ms) | 15–20 FPS at 960 px wide |
| imgsz 320 | ≈ 15–25 detections/s | 20 FPS |
| imgsz 640 | ≈ 6–12 detections/s | 12–20 FPS, boxes lag more |
| PyTorch `.pt` (no export) | roughly half of NCNN | lower |

Expect less when OpenTAKServer is busy on the same Pi (`bft-fpv` profile). The UI shows real
numbers live: video fps, detect fps, ms per inference, and frames per detection. Use
`--benchmark` on the Pi and record the real figure here.

## Detection, tracking and counting

- **Model:** YOLO26n, the Ultralytics nano model, NMS-free and aimed at edge CPUs. You can
  swap it with `FPV_MODEL=yolo11n` or a path. The order of preference is
  `<data>/models/<name>_ncnn_model`, then `<name>.onnx`, then `<name>.pt`. To export:
  `python -m fpv_vision.export [--format ncnn|onnx] [--imgsz 416]`. The size is fixed at export.
- **Classes:** by default only people, vehicles (bicycle, car, motorcycle, bus, train, truck),
  boats and aircraft are reported. Set `FPV_ALL_CLASSES=1` for all 80 COCO classes.
- **Colours:** people are red, vehicles orange, everything else cyan.
- **Adaptive frame-skip:** the detect thread always takes the newest frame, so on a slow CPU
  it skips frames instead of falling behind. The render thread draws the latest boxes on every
  frame, up to `FPV_TARGET_FPS`. If the overlay falls below about 85% of its target, the
  detector inserts a growing pause, shown as "detector backing off N ms", to free CPU for the
  video, and removes it again when there is headroom. The trade-off is that boxes can lag by
  one inference, about 0.1–0.2 s on the Pi.
- **Counting:** "in view" is the current frame. "Unique (tracked)" counts ByteTrack IDs that
  survived at least 3 detections, which filters flicker, and counts each one once. ByteTrack
  has no re-identification, so a person who leaves the frame and comes back, or is hidden for
  a long time, gets a new ID. A looping clip also counts again on every loop. Use
  **Reset counts**.
- **Limits worth teaching:** small objects from altitude are the main weakness of a nano model
  at 416 px (see the aerial sample). Raising `FPV_IMGSZ` to 640 helps at the cost of FPS.
  Tiled inference (SAHI) would help more but is too slow for this CPU. Night, motion blur,
  analog noise and OSD overlays all reduce confidence. The COCO classes come from everyday
  photos, not military vehicles or camouflage, so a fine-tuned model would be needed for
  that.

## Gemini mode

- Optional. Set `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`, default `gemini-3.8-flash`)
  in `/etc/fpv-vision.env`. It uses the official `google-genai` SDK.
- **Describe scene** takes a snapshot and sends the raw frame, with the local detector's
  summary as context, to Gemini, which returns a 1–2 sentence observation such as
  "2 people standing near a pickup by the treeline". Without a key, or if the call fails,
  the button returns the **local summary** instead. That summary comes only from the detector
  boxes, for example "2 people near a car, left of frame; 1 truck right". The UI always labels
  which mode produced the text.
- **Rate limit:** at most one call every `GEMINI_MIN_INTERVAL` seconds (default 10). Calls
  sooner than that return the local summary with a "retry in N s" note.
- **Auto summary:** off by default. In the UI choose every 15 s, 30 s, 1 min or 2 min, or set
  `GEMINI_PERIODIC_SECONDS`. It never runs faster than the rate limit, and it **skips empty
  scenes** so API calls aren't wasted.
- **Privacy:** cloud mode sends images to Google. Use demo footage only.

## Spot reports: ATAK and mesh

The **Send spot report** panel takes these inputs:

- **Location:** an MGRS grid (`30U WB 87427 67248`, 4–10 digits) or `lat, lon`, entered by the
  operator. It is pre-filled from `FPV_DEFAULT_POSITION`, which you can set to the observation
  post or the area of interest.
- **Activity** (free text) and **Unit** (default `unk`).
- A selected gallery snapshot, or a fresh one if none is selected. The Size and Equipment
  fields and the summary come from that snapshot's detections.

It sends two things:

1. **CoT** to `TAK_HOST:TAK_PORT` (default `127.0.0.1:8087`, OpenTAKServer plain TCP): a
   `a-u-G` (unknown ground) marker at the entered position, `how="h-e"` (human-entered), with
   a 1-hour stale time. The remarks hold the full SALUTE report and the summary. If TAK is
   down, the report is queued and retried with backoff for up to 5 minutes. The app never
   crashes because TAK is unavailable.
2. **Mesh** text, under 200 characters, to MQTT `meshbus/tx` on `127.0.0.1:1883` (guest/guest).
   The blufor-tracker bridge forwards it to the Meshtastic radio. fpv-vision **never opens the
   radio's serial port**. Example:

   ```
   SPOTREP FPV-VISION S:5 pax, 3 veh L:30U WB 87427 67248 T:161307ZNOV26 U:unk E:3x car A:pax crossing junction
   ```

   Inbound mesh texts (`meshbus/rx`) appear in the **Mesh inbox** panel, so a patrol can
   acknowledge. If the broker is down, the UI says so and the app carries on.

## Why detections are not geolocated

The CoT marker is placed at the **position the operator typed**, not at a computed location
for the boxed objects. Turning a pixel into a ground coordinate needs:

- the drone's GPS position and **altitude above ground** (telemetry, e.g. MAVLink/MSP/DJI SDK),
- the **camera pose**: gimbal or fixed tilt, plus the aircraft's roll, pitch and yaw, time-synced to each frame,
- the **lens intrinsics** (focal length, distortion) of the actual camera,
- a **terrain/elevation model** to intersect the line of sight with the ground.

An analog FPV feed carries none of this, and a 5" FPV quad usually has a fixed, uncalibrated
camera at a tilt that varies with throttle. Faking a position from video alone would put
confident-looking markers in the wrong place, which is worse than no marker. Adding this would
mean feeding flight-controller telemetry and a calibrated camera model into the pipeline.

## Demo script

About 10 minutes for a section, using a tablet on the Pi's Wi-Fi.

1. **Set the scene (1 min).** Open `http://<pi>:8110`. Point out the header: `LOCAL ONLY (offline)`,
   the model and backend (`yolo26n · ncnn · ByteTrack`) and the TAK and mesh pills. Explain:
   "Everything you see is running on this Raspberry Pi's CPU. No cloud and no AI chip."
2. **Simulate first (2 min).** Leave the source on `Simulate: street crossing`. Show red boxes
   for people, orange for vehicles, and the track IDs. Show the counts table: *in view*
   compared with *unique (tracked)*. Point at the FPS tiles: detect fps compared with video
   fps, and frames per detection. Explain the frame-skip.
3. **Limits (1 min).** Switch to `Simulate: NASA drone pass`. Fewer and smaller detections, and
   almost no people. Discuss altitude, pixels on target, and why "the AI" is not magic.
4. **Live feed (3 min).** Switch to `USB capture /dev/video0` with the FPV goggles' AV-out or
   the GoPro's HDMI going into the dongle. Fly or walk the quad (or the GoPro) past a volunteer
   and a vehicle. Watch the unique count increase once per person. Press **Record** and stop
   after 20 s; the clip appears under **Clips**.
5. **Report (2 min).** Press **Describe scene**: the local summary, or Gemini's if a key is set
   and you have internet. Tap the snapshot in the gallery, type the OP's grid and an activity,
   and press **Send spot report**. Show the marker appearing in ATAK with the SALUTE text in
   its remarks, and the SPOTREP text arriving on a Heltec node.
6. **Discuss (1 min).** Why the marker is at the *entered* grid and not the target (see above).
   What cloud mode would send off the device. How this would plug into a real ISR reporting
   chain.

Reset between groups with **Reset counts**. Snapshots and clips persist in `/var/lib/fpv-vision`.

## Configuration reference

All settings live in `/etc/fpv-vision.env` (see [`deploy/fpv-vision.env.example`](deploy/fpv-vision.env.example)).
CLI flags override them: `--simulate`, `--source`, `--port`, `--model`, `--imgsz`, `--no-detect`,
`--data-dir`, `--no-tak`, `--no-mesh`, `--benchmark N`. Extra flags for the service can go in
`FPV_EXTRA_ARGS`.

| variable | default | meaning |
|---|---|---|
| `FPV_SOURCE` | `simulate` (`auto` in env example) | video source spec |
| `FPV_CAPTURE_WIDTH/HEIGHT` | 1280×720 | requested UVC capture size |
| `FPV_MODEL` / `FPV_IMGSZ` / `FPV_CONF` | `yolo26n` / 416 / 0.30 | detector |
| `FPV_ALL_CLASSES` | 0 | 1 = all COCO classes |
| `FPV_TARGET_FPS` | 20 | overlay/stream cap |
| `FPV_STREAM_WIDTH` / `FPV_JPEG_QUALITY` | 960 / 75 | MJPEG size and quality |
| `FPV_DATA_DIR` | `/var/lib/fpv-vision` | snapshots, clips, models, samples |
| `FPV_MAX_CLIP_SECONDS` | 300 | auto-stop recording |
| `FPV_CALLSIGN` | `FPV-VISION` | CoT sender and mesh prefix |
| `FPV_DEFAULT_POSITION` | – | pre-filled spot report location |
| `TAK_HOST` / `TAK_PORT` / `TAK_ENABLED` | 127.0.0.1 / 8087 / 1 | CoT output |
| `MQTT_HOST/PORT/USER/PASS`, `MESH_ENABLED` | 127.0.0.1 / 1883 / guest / guest / 1 | meshbus |
| `GEMINI_API_KEY` | – | enables cloud mode |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model id |
| `GEMINI_MIN_INTERVAL` / `GEMINI_PERIODIC_SECONDS` | 10 / 0 | rate limit and auto-summary |

Recorded clips are MPEG-4 Part 2 (`mp4v`) `.mp4` files, because the Pi 5 has no hardware H.264
encoder. They play in VLC and most desktop players. Some browsers will download them rather than
play them inline.

## API

| method | path | |
|---|---|---|
| GET | `/stream.mjpg` | annotated MJPEG stream |
| GET | `/frame.jpg` | latest annotated frame |
| GET | `/api/status` | everything the UI shows |
| GET/POST | `/api/sources`, `/api/source {spec}` | list and switch sources |
| POST | `/api/snapshot`, `/api/describe {mode: auto\|local\|gemini}` | snapshot, or snapshot plus summary |
| GET | `/api/snapshots`, `/media/snapshots/<file>` | gallery |
| POST | `/api/record/start`, `/api/record/stop`; GET `/api/clips`, `/media/clips/<file>` | clips |
| POST | `/api/periodic {seconds}` | Gemini auto-summary period (0 = off) |
| POST | `/api/spot-report {position, activity, unit, snapshot?, send_cot, send_mesh}` | spot report |
| POST | `/api/counts/reset` | reset unique counts |

## Development

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The tests use a fake detector and synthetic frames, so they need no hardware, network or
weights. `tests/test_yolo_optional.py` also runs the real model on the street sample if
`data/models` and `data/samples` exist (after the export and `python -m fpv_vision.sample data`).

Layout: `fpv_vision/sources.py` handles capture, `detector.py` YOLO and ByteTrack, and
`pipeline.py` the detect and render threads, recorder and snapshots. `summary.py` builds the
offline scene text, `gemini.py` handles cloud mode, `geo.py` converts MGRS and lat/lon,
`spotrep.py` builds the SALUTE report, `cot.py` the TAK output, `mesh.py` the meshbus
connection, `service.py` the glue, and `web/` the FastAPI app and UI.
