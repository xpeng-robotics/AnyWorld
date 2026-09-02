# Data formats and geometry

## Embodiment-editor metadata

The metadata file is a JSON list. Each item contains:

- <code>image</code>: target robot image, relative to the dataset root;
- <code>edit_image</code>: human-like source images, relative to the root;
- <code>prompt</code>: target-embodiment editing instruction.

## World-model validation manifest

The manifest is either a JSON list or an object with a <code>data</code> list.

| Field | Meaning |
|---|---|
| <code>episode</code> or <code>id</code> | unique sample identifier |
| <code>caption</code> | structured scene/action/embodiment description |
| <code>image_path</code> | edited target-embodiment first frame |
| <code>control_video_path</code> | rendered skeleton action video |
| <code>state_camera_extrinsic_path</code> | NumPy per-frame w2c extrinsics |
| <code>camera_fx/fy/cx/cy</code> | original-coordinate intrinsics |
| <code>camera_orig_width/height</code> | original image dimensions |
| <code>spatial_preprocess</code> | <code>center_crop_resize</code> |

Optional <code>camera_xi</code> and <code>camera_dist</code> carry Mei/fisheye
calibration. Use null for pinhole cameras.

## Temporal and spatial invariants

The video length and its Wan VAE latent length must match the public
model configuration. Their temporal relation is
<code>T_video = 4 × (T_latent - 1) + 1</code>.

RGB, skeleton control, and first-frame VAE inputs must use the same 832×480
center-crop-and-resize transform. Keep intrinsics in the original image
coordinate system; <code>camera_controller.py</code> applies the transform.
Extrinsics are w2c and accept <code>(T, 12)</code>, <code>(T, 3, 4)</code>, or
<code>(T, 4, 4)</code>.
