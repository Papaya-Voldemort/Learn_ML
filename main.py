import os
import json
import urllib.request
import generate_data
import train_model
import fine_tune
from concurrent.futures import ThreadPoolExecutor

def download_coco_images(target_dir="images", num_people=1000, num_objects=200):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    total_target = num_people + num_objects

    # Check existing images
    existing_images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if len(existing_images) >= total_target:
        print(f"✅ Found {len(existing_images)} existing images in '{target_dir}'. Skipping download pipeline.")
        return

    print("🔍 Fetching COCO annotations to select images...")
    json_url = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances_val2017.json")

    try:
        if not os.path.exists(json_path):
            print(f"📥 Downloading annotation metadata ({json_url})...")
            urllib.request.urlretrieve(json_url, json_path)
            print("📦 Metadata download complete.")

        print("📖 Reading and parsing annotations...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find person category ID
        person_cat_id = 1
        for cat in data.get('categories', []):
            if cat.get('name') == 'person':
                person_cat_id = cat.get('id', 1)
                break

        # Map image_id to all categories present in it
        img_to_cats = {}
        for ann in data.get('annotations', []):
            img_id = ann.get('image_id')
            cat_id = ann.get('category_id')
            if img_id is not None and cat_id is not None:
                if img_id not in img_to_cats:
                    img_to_cats[img_id] = set()
                img_to_cats[img_id].add(cat_id)

        # Map image_id to image metadata
        images_dict = {img['id']: img for img in data.get('images', [])}

        person_img_ids = []
        object_only_img_ids = []

        import random
        # Seed for reproducibility of selection
        random.seed(42)

        all_annotated_ids = list(img_to_cats.keys())
        random.shuffle(all_annotated_ids)

        for img_id in all_annotated_ids:
            if img_id not in images_dict:
                continue
            cats = img_to_cats[img_id]
            if person_cat_id in cats:
                person_img_ids.append(img_id)
            else:
                object_only_img_ids.append(img_id)

        selected_people = person_img_ids[:num_people]
        selected_objects = object_only_img_ids[:num_objects]
        selected_ids = selected_people + selected_objects

        print(f"🎯 Selected {len(selected_people)} 'person' images and {len(selected_objects)} 'object-only' images.")
        print(f"📥 Downloading {len(selected_ids)} images concurrently...")

        def download_single_image(img_id):
            img_meta = images_dict[img_id]
            url = img_meta.get('coco_url')
            if not url:
                return
            filename = img_meta.get('file_name')
            dest = os.path.join(target_dir, filename)
            
            # Check if this image already exists to avoid redundant download
            if os.path.exists(dest):
                return

            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"⚠️ Failed to download {url}: {e}")

        # Concurrent downloading using 16 threads
        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(download_single_image, selected_ids)

        print(f"🎉 Download sequence finished. Images saved to '{target_dir}'.")

    finally:
        # Clean up annotation json file
        if os.path.exists(json_path):
            os.remove(json_path)

def main():
    print("=== Phase 1: Preparing Image Dataset ===")
    download_coco_images("images", num_people=1000, num_objects=200)

    print("\n=== Phase 2: Generating Synthetic Character Patches ===")
    generate_data.main()

    print("\n=== Phase 3: Pre-training Base Model ===")
    train_model.main()

    print("\n=== Phase 4: Fine-tuning Model with COCO Real Images ===")
    fine_tune.main(['images', '--epochs', '3', '--max-per-char', '5'])

    print("\n🚀 Pipeline finished successfully! Model is fully trained and fine-tuned.")

if __name__ == "__main__":
    main()