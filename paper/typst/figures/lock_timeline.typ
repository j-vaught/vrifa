// Lock-frames temporal hysteresis schematic.
//
// Per-pixel timeline showing how a pixel becomes 'locked' after
// `lock_frames` consecutive frames of being filled. Pure schematic;
// no real data needed.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let rose      = rgb("#CC2E40")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")
#let b10       = rgb("#ECECEC")

// Sequence of detection state across 12 frames. 1 = "wet" detection
// fired this frame, 0 = no detection.
#let detections = (0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1)
// lock_frames = 3, so the counter must hit 3 before the lock latches.
#let lock-frames = 3

#cetz.canvas({
  import cetz.draw: *

  let cell-w = 0.85
  let cell-h = 0.55
  let n = detections.len()

  // Header.
  content((0, 0.95),
          text(size: 9pt, weight: 700, fill: garnet)[per-pixel state by frame],
          anchor: "west")

  // Frame-index axis above.
  for i in range(n) {
    let x = i * cell-w
    content((x + cell-w / 2, 0.45),
            text(size: 7.5pt, fill: b70)[#(i + 1)])
  }

  // Row 1: detection in this frame.
  for i in range(n) {
    let x = i * cell-w
    let d = detections.at(i)
    rect((x, 0), (x + cell-w, -cell-h),
         fill: if d == 1 { atlantic.lighten(40%) } else { b10 },
         stroke: 0.5pt + b70)
    content((x + cell-w / 2, -cell-h / 2),
            text(size: 8pt, fill: if d == 1 { atlantic } else { b50 },
                 weight: if d == 1 { 700 } else { 400 })[#d])
  }
  content((-0.25, -cell-h / 2),
          text(size: 8pt, fill: b70)[detection],
          anchor: "east")

  // Row 2: counter. Increments while detection stays 1, resets to 0
  // when detection is 0. Capped at lock_frames once latched.
  let counter = 0
  let counters = ()
  for i in range(n) {
    if detections.at(i) == 1 {
      counter = counter + 1
    } else {
      counter = 0
    }
    counters.push(counter)
  }
  for i in range(n) {
    let x = i * cell-w
    let c = counters.at(i)
    rect((x, -cell-h - 0.1), (x + cell-w, -2 * cell-h - 0.1),
         fill: if c >= lock-frames { horseshoe.lighten(40%) }
               else if c > 0 { horseshoe.lighten(75%) }
               else { b10 },
         stroke: 0.5pt + b70)
    content((x + cell-w / 2, -1.5 * cell-h - 0.1),
            text(size: 8pt, fill: if c >= lock-frames { horseshoe.darken(20%) }
                                 else if c > 0 { horseshoe }
                                 else { b50 },
                 weight: if c > 0 { 700 } else { 400 })[#c])
  }
  content((-0.25, -1.5 * cell-h - 0.1),
          text(size: 8pt, fill: b70)[counter],
          anchor: "east")

  // Row 3: locked? Latches True the first frame counter reaches lock_frames.
  let locked = false
  let lockstates = ()
  for i in range(n) {
    if counters.at(i) >= lock-frames {
      locked = true
    }
    lockstates.push(locked)
  }
  for i in range(n) {
    let x = i * cell-w
    let l = lockstates.at(i)
    rect((x, -2 * cell-h - 0.2), (x + cell-w, -3 * cell-h - 0.2),
         fill: if l { rose.lighten(40%) } else { b10 },
         stroke: 0.5pt + b70)
    content((x + cell-w / 2, -2.5 * cell-h - 0.2),
            text(size: 8pt, fill: if l { rose.darken(20%) } else { b50 },
                 weight: if l { 700 } else { 400 })[#if l [✓] else [✗]])
  }
  content((-0.25, -2.5 * cell-h - 0.2),
          text(size: 8pt, fill: b70)[locked?],
          anchor: "east")

  // Latch arrow at the lock point (frame 8 in this example, where
  // counter first reaches 3).
  let latch-i = -1
  for i in range(n) {
    if counters.at(i) >= lock-frames and latch-i == -1 {
      latch-i = i
    }
  }
  if latch-i >= 0 {
    let x = latch-i * cell-w + cell-w / 2
    line((x, -2 * cell-h - 0.18), (x, -2 * cell-h - 0.0),
         stroke: 1.0pt + rose, mark: (start: ">", end: none))
    content((x, -3 * cell-h - 0.45),
            text(size: 8pt, fill: rose, weight: 700)[latches at frame #(latch-i + 1)],
            anchor: "north")
  }
})
