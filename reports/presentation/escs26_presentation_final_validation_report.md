# ESCS'26 Presentation Final Validation Report

**Presentation:** Real-Time PCB Defect Detection for Embedded Visual Inspection
**Final PowerPoint:** `escs26_pcb_defect_detection_presentation_final.pptx`
**Slide count:** 27 total (17 main presentation slides + 10 Q&A backup slides)
**Format:** 16:9 widescreen

## Redesign Summary

The full deck was restyled for a live Zoom conference presentation using Aptos, white/light backgrounds, dark green, teal, blue, and restrained orange accents.

Major redesign work was completed on:

- Slide 5: all six defect examples are now visible beside the dataset split statistics.
- Slides 6-8: the evaluation gate, model settings, and resolution-control logic were enlarged and clarified.
- Slides 9-12: result tables remain editable, exact values are preserved, and the central comparisons are shown with larger direct labels.
- Slides 13-14: the success and failure examples were cropped from the original paper figures without cropping detections, then enlarged into six readable panels.
- Slide 15: the embedded demonstration video is the primary visual, with a compact workflow and deployment-scope panel.
- Slide 17: the previous separate thank-you slide was merged into the takeaway slide to produce the requested 17-slide main flow.
- Slides 18-27: a clearly labeled Q&A backup appendix preserves additional protocol, label-mapping, validation, per-class, fusion, latency, provenance, and environment details without crowding the live talk.

## Typography

- Main title: 36 pt
- Slide titles: 30 pt
- Main result values and takeaways: approximately 19-34 pt
- Body text: generally 16-21 pt
- Tables: 16-18 pt
- Minor section labels and slide numbers: 11-15 pt

Supporting text was shortened into concise labels and presentation-friendly phrases. No scientific result, dataset statistic, model setting, class name, limitation, or deployment claim was removed. The only removed slide was the redundant standalone thank-you slide; its content is retained on Slide 17.

## Scientific Content Audit

The presentation contains the required distinction among:

1. Practical comparison: YOLO11s at 1280 pixels versus RT-DETR-L at 640 pixels.
2. Matched-resolution comparison: YOLO11s and RT-DETR-L at 640 pixels.
3. Separate high-resolution analysis: RT-DETR-L at 1280 pixels.

All required dataset counts, aggregate metrics, per-class metrics, latency values, model names, and six defect class names were found in the final editable PowerPoint. The practical comparison is not described as an architecture-only controlled comparison. Jetson and TensorRT are mentioned only as future work or explicit non-claims.

## Embedded Video Validation

- **Embedded video preserved:** Yes. The MP4 inside the final PPTX is byte-for-byte identical to `reports/presentation/demo_video/demo_hf.mp4`.
- **Poster frame preserved:** Yes. The poster image checksum matches the existing source poster.
- **Self-contained playback:** Yes at the package level. Both PowerPoint media relationships point to the MP4 inside the PPTX; there is no external file or internet relationship.
- **Aspect ratio and crop:** Preserved. The video shape matches the source aspect ratio and has no poster-crop metadata.
- **Playback settings changed:** No. Playback remains on click, volume remains 80%, and no trim points were added.
- **Slide Show test:** Slide 15 was launched in Microsoft PowerPoint Slide Show mode, the normal click was delivered to the video, and the slide show remained running seven seconds later. macOS blocked automated screen capture, so frame-by-frame visual progression could not be recorded by the test script. A final manual click-through on the presenting Mac is still recommended before the conference.
- **Playback controls and overlays:** No text or decorative object overlaps the video. Supporting content is positioned to its right, leaving the video surface and PowerPoint controls unobstructed.

## File Validation

- PowerPoint opened successfully in Microsoft PowerPoint.
- The final PPTX contains 17 main slides followed by 10 backup slides.
- PPTX ZIP/package integrity check passed.
- All slide objects remain inside the 16:9 slide bounds.
- Tables and text remain editable.
- A PDF export will show the video poster frame only; the playable self-contained video remains in the PPTX.

## Presenter Check

Before presenting, open Slide 15 in Slide Show mode, click the video once, confirm that it plays through the 20.83-second demo, and then advance to Slide 16. This is the only remaining human check required because macOS denied automated screen capture during validation.
