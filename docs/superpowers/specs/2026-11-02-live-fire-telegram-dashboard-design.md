# Live Fire Detection and Telegram Dashboard Design

Date: 2026-11-02
Status: design approved in conversation; written spec awaiting user review

## Goal

Turn the existing FireNet Gradio demo into a usable live-monitoring interface. FireNetCNN detects fire or smoke in webcam/video frames, SAM adds region-mask evidence, and Muse Spark reviews the candidate event. The application sends a Telegram alert automatically for Spark-confirmed fire/smoke events and reports video-feed failures as separate operational alerts.

## User flow

1. The operator selects a webcam or video source, frame sampling interval, and alert confidence threshold, then starts monitoring.
2. The app samples frames and runs FireNetCNN locally. It creates an event when a frame is classified as fire or smoke.
3. SAM generates an optional mask overlay for the event frame. SAM output is presented as spatial evidence only, not as a classifier explanation or proof of fire.
4. Muse Spark reviews the candidate frame and event context. If Spark confirms fire or smoke, the application automatically creates and sends a Telegram alert. A high CNN confidence is highlighted as high priority; the selected threshold is visible and configurable.
5. The alert includes a short sampled clip spanning available pre-event and post-event frames, event summary, timestamp, CNN and Spark confidence, SAM evidence, and aggregate evaluation metrics with their dataset/split/sample provenance.
6. If video/webcam ingestion fails, the app reports the operational failure through Telegram when configured and enabled. A Spark, SAM, or clip-generation failure is recorded clearly; it must not fabricate a detection or erase a valid CNN result.

## Dashboard design

Use a field-operations visual language: warm ash surfaces, deep cypress green, ember orange reserved for fire events, restrained telemetry typography, and compact readable status labels. The event timeline is the main visual anchor.

The Gradio dashboard contains:

- A source and controls panel for webcam/video selection, sampling interval, confidence threshold, Spark live-call opt-in, and monitoring start/stop state.
- A live or playback feed beside a timestamped, scrub-able sampled-frame timeline with event markers.
- Event status cards showing event class, per-model confidence, source badges (CNN, Spark, cache, abstention/error), latency, and priority.
- A “Why this alert?” evidence panel linking the exact frame and timestamp to CNN/Spark results, SAM overlay, and any context used. Keep observations, masks, and generated interpretations distinct.
- A Spark chat scoped to the selected event and its supplied frame/evidence. Answers must distinguish observed evidence from interpretation.
- Evaluation cards for accuracy, precision, recall, smoke recall, sample count, latency, and escalation rate. Aggregate metrics must be labeled with their evaluation run/split and never presented as confidence for an individual event.
- Alert history with sent/failed status and a visible record of operational feed failures.

## Telegram and safety behavior

- Positive CNN fire/smoke candidates are sent to Spark for review; only a Spark-confirmed fire/smoke result triggers an automatic incident alert.
- The CNN confidence threshold controls the high-priority designation and is configurable. Spark confirmation remains required for both high- and lower-confidence candidates.
- Real Spark calls and Telegram sending require explicit configuration and live-mode opt-in. Missing credentials or opt-in blocks external calls and is visible in the UI.
- A feed failure is a distinct operational alert, not a fire classification. It must include source and failure reason, and contain no fabricated fire metrics.
- SAM failure does not block a valid Spark-confirmed alert; the alert records that SAM evidence is unavailable.
- Telegram output is text plus an uploaded event clip when available. If clip assembly fails, send available individual before/after frames where possible and disclose the clip failure.
- Never claim continuous analysis or persistent object tracking when processing is sampled and masks are per-frame.

## Evaluation and acceptance

- Preserve the existing held-out test split. Run only a small deterministic, class-balanced subset by default (12 images, configurable seed), and report the exact selected files and per-class counts.
- Evaluate local-only and local-to-Spark using the same subset. Include accuracy, per-class precision/recall/F1, smoke recall, latency, estimated cost, escalation rate, cache/live counts, sample count, split, and seed.
- Write subset results to separate outputs by default. Do not tune thresholds on the test subset, silently substitute other splits, or count failed calls as labels.
- Mock tests cover real-time frame sampling and timeline selection, event window/clip behavior, Spark confirmation, SAM isolation, alert formatting, Telegram feed-failure/incident behavior, UI state updates, and metric provenance.
- Automated tests must not make live API calls, download model weights, send Telegram messages, or overwrite existing results.

## Existing implementation constraints

- The current Gradio UI is image-oriented and runs a single inference on button click; video frame sampling and event-window/clip helpers exist but are not wired into a live monitoring loop.
- Current SAM is local Ultralytics SAM and the code declares text prompts unsupported. Do not claim named text-prompt segmentation unless a compatible backend is explicitly configured and verified.
- Telegram code currently requires manual confirmation and sends only a text message. Implement automatic event-triggered sending through the application while retaining configuration/live-call gates.
- Existing working-tree changes must be preserved; modify only files required for this design.
