import os
import uuid
import folder_paths
import numpy as np

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
)
from pathlib import Path
from comfy_api.input import VideoInput

model_directory = os.path.join(folder_paths.models_dir, "Qwen")
os.makedirs(model_directory, exist_ok=True)


class DownloadAndLoadQwen2_5_VLModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-32B-Instruct",
                        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                        "Qwen/Qwen2.5-VL-72B-Instruct",
                        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
                    ],
                    {"default": "Qwen/Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "8bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
            },
        }

    RETURN_TYPES = ("QWEN2_5_VL_MODEL",)
    RETURN_NAMES = ("Qwen2_5_VL_model",)
    FUNCTION = "DownloadAndLoadQwen2_5_VLModel"
    CATEGORY = "LK节点/QwenVL"

    def DownloadAndLoadQwen2_5_VLModel(self, model, quantization, attention):
        Qwen2_5_VL_model = {"model": "", "model_path": ""}
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Qwen2.5VL model to: {model_path}")
            from modelscope import snapshot_download

            snapshot_download(
                repo_id=model, local_dir=model_path, local_dir_use_symlinks=False
            )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        Qwen2_5_VL_model["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
        )
        Qwen2_5_VL_model["model_path"] = model_path

        return (Qwen2_5_VL_model,)


class Qwen2_5_VL_Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen2_5_VL_model": ("QWEN2_5_VL_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen2_5_VL_Run"
    CATEGORY = "LK节点/QwenVL"

    def Qwen2_5_VL_Run(
        self,
        text,
        Qwen2_5_VL_model,
        video_decode_method,
        max_new_tokens,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
        BatchImage=None,
    ):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28

        # Check if GGUF or HF model
        if Qwen2_5_VL_model.get("is_gguf"):
            return self.run_gguf_model(
                Qwen2_5_VL_model, text, image, video, BatchImage,
                max_new_tokens, min_pixels, max_pixels, total_pixels, seed
            )
        else:
            return self.run_hf_model(
                Qwen2_5_VL_model, text, image, video, BatchImage,
                video_decode_method, max_new_tokens, min_pixels, max_pixels, total_pixels, seed
            )

    def run_hf_model(self, Qwen2_5_VL_model, text, image, video, BatchImage,
                    video_decode_method, max_new_tokens, min_pixels, max_pixels, total_pixels, seed):
        """Original HF Model Logic"""
        processor = AutoProcessor.from_pretrained(Qwen2_5_VL_model["model_path"])

        content = self.prepare_content(image, video, BatchImage, text, min_pixels, max_pixels, total_pixels, seed)

        messages = [{"role": "user", "content": content}]
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen2_5_VL_model["model"].device)
        generated_ids = Qwen2_5_VL_model["model"].generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return (str(output_text[0]),)

    def prepare_content_for_llama_cpp(self, image, video, BatchImage, text, min_pixels, max_pixels, total_pixels, seed):
        """Special content preparation for llama.cpp"""
        content = []

        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                # llama.cpp expects file:// URLs
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": uri}  # Keep file:// prefix!
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": path}  # Keep file:// prefix!
                        }
                    )

        if video is not None:
            print("⚠️ Video support in llama.cpp might be limited")
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": uri}  # Keep file:// prefix!
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": path}  # Keep file:// prefix!
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        return content

    def run_gguf_model(self, Qwen2_5_VL_model, text, image, video, BatchImage,
                    max_new_tokens, min_pixels, max_pixels, total_pixels, seed):
        """GGUF Model Logic with llama.cpp"""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            if "llm_instance" not in Qwen2_5_VL_model:
                model_path = Qwen2_5_VL_model["model_path"]
                mmproj_path = self.find_mmproj_path(model_path)

                print(f"🚀 Loading GGUF model: {model_path}")
                print(f"📁 Using mmproj: {mmproj_path}")

                chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                llm = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    verbose=False
                )
                Qwen2_5_VL_model["llm_instance"] = llm
            else:
                llm = Qwen2_5_VL_model["llm_instance"]

            # Special content preparation for llama.cpp
            content = self.prepare_content_for_llama_cpp(image, video, BatchImage, text, min_pixels, max_pixels, total_pixels, seed)

            # Debug: Show what's being sent to llama.cpp
            print(f"📤 Sending to llama.cpp: {content}")

            messages = [{"role": "user", "content": content}]

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                stream=False
            )

            output_text = response["choices"][0]["message"]["content"]
            return (str(output_text),)

        except Exception as e:
            import traceback
            print(f"❌ Detailed GGUF error: {traceback.format_exc()}")
            return (f"Error in GGUF model: {str(e)}",)

    def find_mmproj_path(self, model_path):
        """Automatically find mmproj path - improved version"""
        import os
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)

        # Basename without extension for pattern matching
        base_name = model_name.replace(".gguf", "").replace(".GGUF", "")

        # Try different mmproj naming patterns
        possible_names = [
            f"{base_name}.mmproj.gguf",
            f"{base_name}.mmproj-f16.gguf",
            f"{base_name}-mmproj.gguf",
            f"{base_name}.mmproj-F16.gguf",
            "Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf",  # Your specific case
            "Qwen2.5-VL-7B-Instruct.mmproj.gguf",
            "qwen2.5-vl-7b-instruct.mmproj.gguf",
        ]

        for name in possible_names:
            mmproj_path = os.path.join(model_dir, name)
            # Check if file exists (including symlinks)
            if os.path.isfile(mmproj_path) or os.path.islink(mmproj_path):
                # Check if symlink target exists
                if os.path.islink(mmproj_path):
                    real_path = os.path.realpath(mmproj_path)
                    if os.path.exists(real_path):
                        return real_path
                    else:
                        continue
                return mmproj_path

        # Debug: Show available files in directory
        available_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        print(f"🔍 Available GGUF files in {model_dir}:")
        for f in available_files:
            print(f"   - {f}")

        raise FileNotFoundError(f"Could not find mmproj file for {model_path}. Available files: {available_files}")

    def prepare_content(self, image, video, BatchImage, text, min_pixels, max_pixels, total_pixels, seed):
        """Prepare content for both model types"""
        content = []

        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        return content


class Qwen2_5_VL_Run_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "system_text": ("STRING", {"default": "", "multiline": True}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen2_5_VL_model": ("QWEN2_5_VL_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen2_5_VL_Run_Advanced"
    CATEGORY = "LK节点/QwenVL"

    def Qwen2_5_VL_Run_Advanced(
        self,
        system_text,
        text,
        Qwen2_5_VL_model,
        video_decode_method,
        max_new_tokens,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
        BatchImage=None,
    ):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28

        processor = AutoProcessor.from_pretrained(Qwen2_5_VL_model["model_path"])

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen2_5_VL_model["model"].device)
        generated_ids = Qwen2_5_VL_model["model"].generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return (str(output_text[0]),)

class Qwen2_5_VL_Run_GGUF:
    def __init__(self):
        self.llm_instance = None
        self.current_model_path = None
        self.current_mmproj_path = None
        self.current_n_ctx = None
        self.current_n_gpu_layers = None

    @classmethod
    def INPUT_TYPES(s):
        # Get models from text_encoders folder (like CLIPLoaderGGUF)
        gguf_files = []
        gguf_files += folder_paths.get_filename_list("clip")  # Normal CLIP files
        gguf_files += folder_paths.get_filename_list("clip_gguf")  # GGUF files
        gguf_files = [f for f in gguf_files if f.endswith('.gguf')]
        gguf_files = sorted(gguf_files)

        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "model_name": (gguf_files,),  # Dropdown Model Selection
                "text": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 131072}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run_gguf"
    CATEGORY = "LK节点/QwenVL"
    TITLE = "Qwen2.5-VL Run (GGUF)"

    def run_gguf(self, model_name, text, max_tokens, temperature, top_p, top_k,
                n_ctx, n_gpu_layers, seed, image=None, video=None, BatchImage=None):

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            # 1. Load GGUF model
            model_path = folder_paths.get_full_path("clip", model_name)
            mmproj_path = self.gguf_mmproj_loader(model_path)

            # Check if we need to create a new instance
            needs_new_instance = (
                self.llm_instance is None or
                self.current_model_path != model_path or
                self.current_mmproj_path != mmproj_path or
                self.current_n_ctx != n_ctx or
                self.current_n_gpu_layers != n_gpu_layers
            )

            if needs_new_instance:
                # Cleanup old instance if exists
                if self.llm_instance is not None:
                    try:
                        # Proper cleanup of llama.cpp instance
                        del self.llm_instance
                    except:
                        pass

                # Create new instance
                chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                self.llm_instance = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )

                # Update cache
                self.current_model_path = model_path
                self.current_mmproj_path = mmproj_path
                self.current_n_ctx = n_ctx
                self.current_n_gpu_layers = n_gpu_layers

            # 2. Prepare content
            content = self.prepare_content_for_llama_cpp(image, video, BatchImage, text, seed)

            messages = [{"role": "user", "content": content}]

            # 3. Text generation with proper parameters
            response = self.llm_instance.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=False
            )

            output_text = response["choices"][0]["message"]["content"]
            return (str(output_text),)

        except Exception as e:
            return (f"Error in GGUF model: {str(e)}",)

    def gguf_mmproj_loader(self, path):
        """Find mmproj file for GGUF text encoder using city96's compatible logic"""
        import os
        import logging
        import re

        logging.info("Attempting to find mmproj file for text encoder...")

        # Get base name without quant suffix (same as city96's logic)
        tenc_fname = os.path.basename(path)
        tenc = os.path.splitext(tenc_fname)[0].lower()
        tenc = self.strip_quant_suffix(tenc)

        # Find matching mmproj files (same logic as city96)
        target = []
        root = os.path.dirname(path)

        for fname in os.listdir(root):
            name, ext = os.path.splitext(fname)
            if ext.lower() != ".gguf":
                continue
            if "mmproj" not in name.lower():
                continue
            if tenc in name.lower():
                target.append(fname)

        if len(target) == 0:
            logging.error(f"Error: Can't find mmproj file for '{tenc_fname}' (matching:'{tenc}')!")
            raise FileNotFoundError(f"Could not find mmproj file for {tenc_fname}")

        if len(target) > 1:
            logging.warning(f"Multiple mmproj files found for text encoder '{tenc_fname}', using first match: {target[0]}")

        logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
        mmproj_path = os.path.join(root, target[0])

        return mmproj_path

    def strip_quant_suffix(self, name):
        """
        Exact copy of city96's quant suffix stripping logic from loader.py
        Uses regex pattern to remove quantization suffixes like -q4_k_m, .q8_0, etc.
        """
        import re
        pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            name = name[:match.start()]
        return name

    def prepare_content_for_llama_cpp(self, image, video, BatchImage, text, seed):
        """Special content preparation for llama.cpp"""
        content = []

        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": uri}
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": path}
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": uri}
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": path}
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        return content

class BatchImageLoaderToLocalFiles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("BatchImage",)
    RETURN_NAMES = ("BatchImage",)
    FUNCTION = "BatchImageLoaderToLocalFiles"
    CATEGORY = "Qwen2_5-VL"

    def BatchImageLoaderToLocalFiles(self, **kwargs):
        images = list(kwargs.values())
        image_paths = []

        for idx, image in enumerate(images):
            unique_id = uuid.uuid4().hex
            image_path = (
                Path(folder_paths.temp_directory) / f"temp_image_{idx}_{unique_id}.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(
                np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            )
            img.save(os.path.join(image_path))

            image_paths.append(f"file://{image_path.resolve().as_posix()}")

        return (image_paths,)


def temp_video(video: VideoInput, seed):
    unique_id = uuid.uuid4().hex
    video_path = (
        Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(
        os.path.join(video_path),
        format="mp4",
        codec="h264",
    )

    uri = f"{video_path.as_posix()}"

    return uri


def temp_image(image, seed):
    unique_id = uuid.uuid4().hex
    image_path = (
        Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    img.save(os.path.join(image_path))

    uri = f"file://{image_path.as_posix()}"

    return uri


def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for Nth_count in range(num_counts):
        img = Image.fromarray(
            np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
        unique_id = uuid.uuid4().hex
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}_{unique_id}.png"
        img.save(os.path.join(image_path))

        image_paths.append(f"file://{image_path.resolve().as_posix()}")

    return image_paths


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQwen2_5_VLModel": DownloadAndLoadQwen2_5_VLModel,
    "Qwen2_5_VL_Run": Qwen2_5_VL_Run,
    "Qwen2_5_VL_Run_Advanced": Qwen2_5_VL_Run_Advanced,
    "Qwen2_5_VL_Run_GGUF": Qwen2_5_VL_Run_GGUF,  # ← ADD NEW NODE
    "BatchImageLoaderToLocalFiles": BatchImageLoaderToLocalFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQwen2_5_VLModel": "DownloadAndLoadQwen2.5_VLModel",
    "Qwen2_5_VL_Run": "Qwen2.5_VL_Run",
    "Qwen2_5_VL_Run_Advanced": "Qwen2.5_VL_Run_Advanced",
    "Qwen2_5_VL_Run_GGUF": "Qwen2.5-VL Run (GGUF)",  # ← DISPLAY NAME
    "BatchImageLoaderToLocalFiles": "BatchImageLoaderToLocalFiles",
}