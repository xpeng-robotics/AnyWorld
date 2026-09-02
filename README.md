# AnyWorld: Factorized Egocentric World Models for Cross-Embodiment Generalization

[**Paper**](https://arxiv.org/abs/2608.29242) · [**Project Page**](https://xpeng-robotics.github.io/anyworld/) · [**中文说明**](README_zh-CN.md)



https://github.com/user-attachments/assets/0b01d6c4-f3cc-409c-aec7-90c116c74d6a





## Highlights

- **Factorized cross-embodiment recomposition.** Independently control action, camera, embodiment, and scene context to turn human interaction into robot-native rollouts.
- **Unpaired human-to-robot grounding.** Learn transferable interaction structure from egocentric human videos and ground it to target robot embodiments without paired human–robot clips.
- **Targeted intervention generation.** The same factorized generator supports both **broad robot-native experience scaling** across actions, viewpoints, embodiments, and scenes, and **targeted intervention for policy gaps** by constructing missing robot-native states and matched counterfactual instruction–action pairs.

## Code

The public release contains:

- `image_editing/` — reverse pseudo-pair construction and embodiment-editor training/inference.
- `world_model/` — factorized action-camera-embodiment world-model inference.
- `scripts/` — image-editor training and world-model inference launchers.
- `docs/` — input formats, geometry conventions, and external model-weight layout.

### Image editing

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .

cd /path/to/AnyWorld
pip install -r image_editing/requirements.txt
bash scripts/train_image_editor.sh
```

Inference:

```bash
python image_editing/infer.py \
  --input /path/to/images_or_videos \
  --output-dir outputs/edited_first_frames \
  --checkpoint /models/AnyWorld-ImageEditor/anyworld_image_editor.safetensors \
  --base-model-path /models/Qwen-Image-Edit-2511 \
  --diffsynth-repo /path/to/DiffSynth-Studio
```

### World-model inference

AnyWorld releases the adapted world-model Transformer separately from the Wan2.1 Combined-Control base components required by the inference pipeline.

```bash
pip install -e world_model

python world_model/scripts/infer.py \
  --base-model-path /models/Wan2.1-I2V-14B-480P-Combined-Control-Diffusers \
  --anyworld-model-path /models/AnyWorld-WorldModel \
  --validation-file validation.json \
  --output-dir outputs/world_model
```

See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) and [`docs/MODEL_WEIGHTS.md`](docs/MODEL_WEIGHTS.md) for input and weight conventions.

## Acknowledgements

Built on [FastVideo](https://github.com/hao-ai-lab/FastVideo), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), and [Wan2.1](https://github.com/Wan-Video/Wan2.1).

## Citation

```bibtex
@article{chen2026anyworld,
  title   = {AnyWorld: Factorized Egocentric World Models for Cross-Embodiment Generalization},
  author  = {Chen, Cheng and Bai, Jerry and Wei, Jiacheng and Chen, Boyu and
             Zheng, Xiaoji and Wu, Fan and Yang, Minghao and Chen, Tianrun and
             Li, Ruibo and Yue, Xiaoyu and Guo, Xiaoyang and Ge, Yixiao and
             Lin, Guosheng and Liu, Fayao},
  journal = {arXiv preprint arXiv:2608.29242},
  year    = {2026}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Modified upstream code retains its original license notices.
