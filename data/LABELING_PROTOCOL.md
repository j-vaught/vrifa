# VRIFA Labeling Protocol

Goal. Produce a 55-frame ground-truth set that locks the quantitative agreement claims in the paper. The frames are already extracted, one polygon per frame is sufficient, and the export format the paper expects is COCO JSON.

## What you are labeling

Resin-wetted (wet) regions in VARTM infusion video. The wet region is the area where resin has reached the fabric, viewed through the transparent vacuum bag. It appears as a darker, more saturated zone advancing across the laminate.

## What counts as wet

- The leading dark front and everything behind it that has clearly transitioned from dry fabric to resin-saturated fabric.
- Pixels where the fabric weave is visibly wetted-out, even if the local color is muddled by lighting.
- Wet zones that have grown and merged behind the front.

## What does *not* count as wet

- Specular reflections from the lab lights on the bag surface. Bright spots are not wet.
- Vacuum-bag wrinkles and creases that pre-date the front. The crease is the same brightness before and after the resin arrives.
- Shadows from rigging, clamps, or operators standing nearby.
- Fabric weave shadows that pre-date the front. If the dark patch was already dark in the first frame, it is not wet.
- Resin that has pooled in the runner / inlet line outside the laminate boundary.

When in doubt, scrub forward a few frames and check whether the candidate region's brightness changes monotonically. Real wetting darkens once and stays dark. Reflections come and go. Wrinkles oscillate.

## Labeling aid: the dry reference frame

For each of the eleven samples, the first video frame is saved at `data/label_aids/<sample>__dry_reference.png`. That frame shows what the laminate looks like *before any resin arrives*. Open the relevant dry reference in a second browser tab while you label the corresponding anchor frames.

Why this matters. A pixel that looks dark in the anchor frame may be:

- A wrinkle, a sealant tape edge, or a fabric weave shadow that was *already dark* in the dry reference. These are not wet.
- Resin that has actually arrived and changed the local appearance. These are wet.

Comparing against the dry reference disambiguates the two in seconds. It is not a substitute for the scrub-forward test (which checks monotonicity over time), but it is a faster first pass.

What we deliberately do **not** provide as a labeling aid is the algorithm's own delta heatmap or predicted mask. Looking at those would bias your label toward what the algorithm already sees, and the IoU we measure against your label would no longer be an honest agreement number. The dry reference shows you the *input* signal, which is fair game; the heatmap shows you the *output*, which would be tautological.

## Tool

Use [makesense.ai](https://www.makesense.ai/). It is browser-only, free, and exports COCO out of the box.

1. Open `https://www.makesense.ai/` in Chrome.
2. Click *Get Started*.
3. Drop every PNG inside `data/label_frames/` into the upload zone. The folder holds 55 PNGs in a flat layout with `<sample_slug>__frame_<index>.png` naming. Filenames are unique across samples, so makesense.ai accepts the whole drop without collisions. Drag the files (not the folder) if your browser does not preserve nested paths.
4. Choose *Object Detection*.
5. Add a single label class named exactly `wet`. No other classes; the algorithm produces one binary mask.
6. For each image, draw a polygon around the wet region following the rules above. One polygon per image is enough; if the wet region is genuinely disconnected (rare), draw multiple.
7. When all 55 images are labeled, click *Actions → Export Annotations → Single file in COCO JSON format*.
8. Save the export as `data/labels_55.json`. The filename is checked by the agreement script, so use that exact path.

## Time budget

Plan for **5–10 hours total** across the 55 frames. Cleaner frames (input_1 and the early-fill positions of input_4 through input_11) take 3–5 minutes each. Late-fill frames where the wet region hugs the laminate boundary and pinches around inlet lines can take 10–15 minutes. Most labelers settle around 6–8 hours once the rhythm clicks.

You do not have to do all 55 in one sitting. The makesense.ai project state persists in browser local storage; refreshing or closing the tab is fine if you keep using the same browser.

## Stopping early

If after the first 11 frames (one per sample, the mid-fill 50 % position) you feel the dataset is already showing what you want, stop and tell me. The agreement script tolerates partial label sets and the per-sample breakdown still works at smaller counts. The thesis sentence in the paper accommodates 33-, 55-, and 110-frame labeled subsets; only the "across N-frame" wording changes.

If you go past 55 voluntarily, that is fine too. More labels improve the bootstrap CIs. Anything labeled lands.

## Edge cases worth flagging in your review

- Samples where the inlet runner is visibly resin-filled but the laminate-side has not yet wetted out. Treat the laminate as the only labelable area; ignore inlet pooling.
- Frames where the operator's hand or a sensor cable enters the field. Label only the visible wet region; do not extend the polygon under occlusions.
- Frames at the 5 % position of a slow video where almost nothing has happened. An empty polygon-set for those frames is allowed and reasonable.

## Sanity check before sending the JSON

Open `data/labels_55.json` in a text editor and confirm:

- the `images` array has 55 entries (or whatever subset count you chose),
- every `image_id` referenced in `annotations` resolves to one of the `images` entries,
- there is exactly one `categories` entry, with `name: "wet"`.

If any of those fails, re-export from makesense.ai before saving. The agreement script gives a useful error if the file is malformed but a clean export saves a round-trip.

## Author

J.C. Vaught, May 2026.
