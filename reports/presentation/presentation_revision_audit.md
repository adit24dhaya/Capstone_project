# ESCS'26 Presentation Revision Audit

> Historical audit of the first revision cycle. The deliverable that supersedes
> this review is `escs26_pcb_defect_detection_presentation_final.pptx`, containing
> 17 main presentation slides and 10 Q&A backup slides. See
> `escs26_presentation_final_validation_report.md` for the final validation.

Decks checked:

- Commented deck: `escs26_pcb_defect_detection_presentation  -  with comments.pptx`
- Revised deck: `escs26_pcb_defect_detection_presentation_revised.pptx`
- Final deck with embedded demo: `escs26_pcb_defect_detection_presentation_revised_with_demo.pptx`

## Summary

The revised deck has 18 slides, compared with 17 slides in the commented version. The available comments were stored in speaker notes rather than PowerPoint comment XML. The revised deck addresses the extracted comment themes: model explanation, readability/framing, resolution fairness, mAP explanation, hard-class interpretation, qualitative-audit clarification, formula clarification, deployment demo, and a separate closing/thank-you slide.

The final deck with demo embeds `reports/presentation/demo_video/demo_hf.mp4` directly into slide 15. Verification found `ppt/media/media1.mp4` inside the PowerPoint archive, so the video is packaged with the deck.

## Comment-by-Comment Check

| Original slide | Extracted comment/theme | Revised status |
|---:|---|---|
| 3 | Explain what YOLO and RT-DETR-L mean. | Addressed in revised speaker notes on slides 3 and 7. |
| 4 | Present paper as a whole, not accepted vs camera-ready; increase small text. | Addressed by reframing slide 4 as integrated contributions. |
| 5 | Some text too small. | Revised dataset slide uses cleaner summary framing; manual final visual check still recommended in PowerPoint. |
| 6 | Some text too small. | Revised protocol slide simplifies the protocol summary. |
| 7 | Explain differences among models. | Addressed in revised slide 7 and notes. |
| 8 | Discuss final version and compare 1280 vs 640. | Addressed by revised slide 8, now titled Experiment Design, covering the resolution-control issue. |
| 9 | Explain mAP50 and mAP50-95. | Addressed in slide 9 notes. |
| 12 | Highlight harder classes and explain Mouse bite/Spur. | Addressed in slide 12 notes and per-class interpretation. |
| 13 | Explain qualitative audit and whether it is class balancing improvement. | Addressed in slide 13 notes: representative detections, not before/after class balancing. |
| 14 | Explain where the formula is used. | Addressed in slide 14 notes: the miss-weighted error score selects failure examples only. |
| 15 | Add a short recorded demo. | Addressed in final deck with embedded 20.8-second Hugging Face demo video. |
| 17 | Some text too small; thank-you could be separate. | Addressed by separating the closing into slides 17 and 18. |

## Demo Slide Update

Slide 15 now includes:

- Embedded video: YOLO11s-1280 Hugging Face Space demo.
- Demo flow: settings, sample selection, detection, class counts, result table, before/after output, exports.
- Notes clarifying that the online demo runs YOLO11s only and is not the paper's YOLO11s vs RT-DETR-L experiment.
- Notes clarifying that CPU Space latency is not the V100 benchmark latency reported in the paper.

## Remaining Manual Checks

Before presenting, open the final deck in PowerPoint and confirm:

1. Slide 15 video plays when clicked.
2. Presenter View shows the speaker notes for slide 15.
3. Font sizes look readable on slides 5, 6, and 17 at projector/Zoom resolution.
4. The final deck opens without media warnings.
