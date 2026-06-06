import os
import sys

def push_model():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("The 'huggingface_hub' package is not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # configure repo information
    username = "Papaya-Voldemort"
    repo_name = "image-to-ascii"
    
    repo_id = f"{username}/{repo_name}"
    model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ascii_cam_model")

    print(f"📦 Preparing to upload folder: '{model_folder}'")
    print(f"🚀 Target Hugging Face Repo:  'https://huggingface.co/{repo_id}'")
    
    # check username
    if username == "YOUR_HF_USERNAME":
        print("\nPlease edit this script first to replace 'YOUR_HF_USERNAME' with your actual username.")
        sys.exit(1)

    # initialize api
    api = HfApi()

    try:
        print("\nCreating repository on Hugging Face (if it doesn't exist)...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("Repository verified/created.")

        print("\nUploading model folder contents to the Hub...")
        # upload folder contents
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
