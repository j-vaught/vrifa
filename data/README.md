# VRIFA labels

The file `labels.json` is a COCO-style annotation file for inputs `1-14`. Category `1` is `Wet` and category `2` is `Dry`.

Standard COCO importers should load `Dry` as a separate class. Binary wet-mask loaders should treat `Wet` polygons as positive regions and subtract `Dry` polygons from them. The dry category is currently used for the small dry spot in `input_13__frame_000905.png`.
