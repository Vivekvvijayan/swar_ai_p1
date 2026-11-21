import os
import sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    print("Error: HF_TOKEN environment variable is not set. Please add HF_TOKEN in GitHub repository secrets.", file=sys.stderr)
    sys.exit(1)

username = os.environ.get("HF_USERNAME", "Vivekvvijayan")
space_name = os.environ.get("HF_SPACE_NAME", "vv_carnatic_raga_ai")
repo_id = f"{username}/{space_name}"

print(f"Uploading repository to Hugging Face Space: {repo_id}...")

try:
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git/**",
            ".github/**",
            "generated_tunes/**",
            "venv/**",
            "__pycache__/**",
            "*.pyc"
        ]
    )
    print(f"Successfully uploaded all files (including model checkpoints) to https://huggingface.co/spaces/{repo_id}!")
except Exception as e:
    print(f"Failed to upload to Hugging Face: {e}", file=sys.stderr)
    sys.exit(1)
