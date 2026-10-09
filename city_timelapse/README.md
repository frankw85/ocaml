# Aldermouth: city growth timelapse in Blender

A procedural timelapse of a fictional river city growing from an Iron Age
ford to a modern metropolis, in the style of "city through the ages" videos.

## Files

- `timeline.py`: years, built-up radius, era names, population and events.
  Edit this to change the story; both the 3D scene and the captions read it.
- `build_scene.py`: builds the scene with `bpy` and renders frames in Cycles.
- `compose.py`: draws the captions on the frames and encodes the MP4.

## How it works

Every building, road segment and ground vertex gets a `birth` year computed
from its distance to the old core (stretched along the river, noisy, slower on
the south bank, plus a few outlying villages). A `Growth` Geometry Nodes
modifier deletes faces whose `birth` is later than the current year, and a
keyframed Value node feeds that year. Building colour shifts with age:
terracotta, then brick, then grey suburbs, with glass towers in the core after
1965. The camera pulls back as the built-up radius grows.

## Run

```sh
pip install bpy            # Blender as a Python module (5.x, Python 3.13)
python3 build_scene.py --out frames --save aldermouth.blend
python3 compose.py --frames frames --out aldermouth.mp4
```

Useful flags for `build_scene.py`: `--res 1920 1080`, `--samples 32`,
`--frames 1 200`, `--still 150 600` (test frames only).

The saved `.blend` opens in regular Blender, so you can tweak materials,
camera or lighting by hand and render from the GUI.
