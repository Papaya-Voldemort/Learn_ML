#!/usr/bin/env python3
"""
Hugging Face Hub Upload Template
================================

This is a template script to push your trained model to the Hugging Face Model Hub.
You will need:
1. The huggingface_hub Python library installed: `pip install huggingface_hub`
2. A Hugging Face account and a User Access Token (with Write permissions).
3. To login locally via CLI: `huggingface-cli login` OR set the token in your environment.
"""

import os
import sys

def push_model():
    # TODO: Install huggingface_hub if you haven't already
    # pip install huggingface_hub
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ The 'huggingface_hub' package is not installed.")
        print("💡 Run: pip install huggingface_hub")
        sys.exit(1)

    # TODO: Fill in your Hugging Face username and the desired model repo name
    username = "YOUR_HF_USERNAME"  # e.g., "elinelson"
    repo_name = "ascii-camera-model"  # e.g., "ascii-camera-onnx"
    
    repo_id = f"{username}/{repo_name}"
    model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ascii_cam_model")

    print(f"📦 Preparing to upload folder: '{model_folder}'")
    print(f"🚀 Target Hugging Face Repo:  'https://huggingface.co/{repo_id}'")
    
    # Simple validation check
    if username == "YOUR_HF_USERNAME":
        print("\n⚠️  Please edit this script first to replace 'YOUR_HF_USERNAME' with your actual username.")
        sys.exit(1)

    # Initialize the API
    api = HfApi()

    try:
        print("\nCreating repository on Hugging Face (if it doesn't exist)...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("✅ Repository verified/created.")

        print("\nUploading model folder contents to the Hub...")
        # Uploads all files in ascii_cam_model (model.onnx, model.keras)
        future = api.upload_folder(
            folder_path=model_folder,
            repo_id=repo_id,
            repo_type="model",
        )
        print("🎉 Upload successful! Your model is now live on Hugging Face Hub.")
        print(f"🔗 View here: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ An error occurred during upload: {e}")
        print("💡 Make sure you ran 'huggingface-cli login' first or have HF_TOKEN set in your environment.")

if __name__ == "__main__":
    push_model()
