# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import dataclasses
from typing import Literal

from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools
import sys

from uno.flux.pipeline import UNOPipeline, preprocess_ref


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = 512
    num_steps: int = 25
    guidance: float = 5
    seed: int = -1
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'


    
# Your main inference function.
def main(args: InferenceArgs):
    accelerator = Accelerator()


    assert args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
    
    with open("/home/ubuntu/perpetualVLMData/run_events.json", "rt") as f:
        data_dicts = json.load(f)
    data_root = "/home/ubuntu/perpetualVLMData/faces"

    for i, data_dict in enumerate(data_dicts):
        # if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
        #     continue

        # Case event 1
        img_path = str(data_dict["idx"]) + "_0.jpg"
        ref_imgs = [
            Image.open(os.path.join(data_root, img_path)).convert("RGB")
        ]
        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]
        image_gen = pipeline(
            prompt=data_dict["event1"]["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            ref_imgs=ref_imgs,
            pe=args.pe,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        output_file = data_dict.get("save_path", 
                            os.path.join("/home/ubuntu/perpetualVLMData/faces", data_dict["event1"]["image"]))
        image_gen.save(output_file)

        # Case event 2
        img_path = str(data_dict["idx"]) + "_0.jpg"
        ref_imgs = [
            Image.open(os.path.join(data_root, img_path)).convert("RGB")
        ]
        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]
        image_gen = pipeline(
            prompt=data_dict["event2"]["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            ref_imgs=ref_imgs,
            pe=args.pe,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        output_file = data_dict.get("save_path", 
                            os.path.join("/home/ubuntu/perpetualVLMData/faces", data_dict["event2"]["image"]))
        image_gen.save(output_file)

# Updated main block to accept a list of annotations.
if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    
    pipeline = UNOPipeline(
        "flux-dev",
        "cuda",
        False,
        True,
        512
    )
    # Check if a JSON file is provided on the command line.
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_path = os.path.abspath(sys.argv[1])
        with open(json_path, "rt") as f:
            data = json.load(f)
            
        # If the JSON file is a list of annotations.
        if isinstance(data, list):
            # Create a default instance of InferenceArgs.
            defaults = InferenceArgs()
            for idx, annotation in enumerate(data):
                # Merge the defaults with the annotation-specific values.
                merged_args = dataclasses.asdict(defaults)
                merged_args.update(annotation)
                args = InferenceArgs(**merged_args)
                print(f"Processing instance {idx + 1}...")
                main(args)
        else:
            # Otherwise, treat it as a single JSON object.
            args = parser.parse_json_file(json_file=json_path)[0]
            main(args)
    else:
        # If no JSON file is provided, fall back to parsing command-line arguments.
        args = parser.parse_args_into_dataclasses()[0]
        main(args)