# Findings

Durable, versioned record of what has actually been measured. Tracked in git on purpose:
`eval_output/` is gitignored and gets cleared, `runs/` holds scalars but no interpretation,
and commit messages are searchable but not browsable. Anything that took GPU time to learn
and would cost GPU time to re-learn belongs here.

## One file per training direction

Direction is named `<source>-<target>`, where `world` is the recurrent trunk:

| file | direction | meaning |
|---|---|---|
| `world-voice.md` | text -> voice | speech synthesis (formerly "world-tts") |
| `world-image.md` | text -> image | image synthesis |
| `world-text.md` | text -> text | continuation / language modelling |
| `voice-world.md` | voice -> text | transcription |
| `image-world.md` | image -> text | captioning |
| `cross-modal.md` | — | principles that hold across directions, and only those |

A finding goes in the direction it was MEASURED in. Promote to `cross-modal.md` only after
it has been replicated in a second direction — that file is for things with evidence from
more than one modality, not for things that feel general.

## Entry format

Every entry carries a status, because the expensive failure in this project has been acting
on a conclusion that was later invalidated:

- **ESTABLISHED** — measured, with the measurement quoted, and the protocol known-good.
- **OPEN** — measured but confounded, underpowered, or single-run. Say what would settle it.
- **RETRACTED** — was believed, now known false. **Never delete these.** A wrong conclusion
  that quietly disappears is worse than one struck through, because the old version is what
  people remember and there is nothing to contradict it. Keep the original claim, the reason
  it was wrong, and the date.

Include the numbers inline. "Text conditioning is weak" is not a finding; "text-attributed
fraction 0.029 at mask ratio 0.25 vs AR's 0.305 at matched step 23000" is.

Note the protocol when it is load-bearing — teacher-forced vs free-running, matched step,
seed count, n. Several entries below exist because a protocol detail was wrong and nobody
had written down what the protocol was.
