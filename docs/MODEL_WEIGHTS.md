# Model weights

Large model weights are intentionally stored outside the Git repository.

## Embodiment editor

~~~text
Qwen-Image-Edit-2511/
├── processor/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/

AnyWorld-ImageEditor/
└── anyworld_image_editor.safetensors
~~~

Pass the Qwen base directory with <code>--base-model-path</code> and the AnyWorld
editor weight file with <code>--checkpoint</code>.

## World model

~~~text
Wan2.1-I2V-14B-480P-Combined-Control-Diffusers/
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
├── transformer/
├── vae/
└── ...

AnyWorld-WorldModel/
└── transformer/
    ├── config.json
    └── model shards / index
~~~

For world-model inference, pass the Wan directory with <code>--base-model-path</code> and the AnyWorld directory with <code>--anyworld-model-path</code>. The Wan directory supplies the base pipeline components; the AnyWorld directory supplies the adapted Transformer weights.

When using <code>scripts/infer_world_model.sh</code>, set <code>BASE_MODEL_PATH</code> and <code>ANYWORLD_MODEL_PATH</code> respectively.
