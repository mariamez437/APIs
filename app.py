from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from enum import Enum
from pydantic import BaseModel, validator
import shutil
import uuid
import os
from PIL import Image
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

app = FastAPI()

IMAGE_FOLDER = "static/images_db"
UPLOAD_FOLDER = "uploads"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


class MatchType(str, Enum):
    text = "text"
    image = "image"
    both = "both"


class MatchTextData(BaseModel):
    match_type: MatchType
    lost_brand: str = ""
    lost_color: str = ""
    lost_government: str = ""
    lost_center: str = ""
    lost_street: str = ""
    lost_contact: str = ""
    found_brand: str = ""
    found_color: str = ""
    found_government: str = ""
    found_center: str = ""
    found_street: str = ""
    found_contact: str = ""

    @validator("*", pre=True)
    def allow_numbers_in_fields(cls, v):
        if isinstance(v, str) and len(v.strip()) == 0:
            return ""
        return v

    @validator("lost_street", "found_street")
    def validate_street(cls, v):
        if not isinstance(v, str):
            raise ValueError("Street must be a string.")
        return v.strip()


attributes = {
    "brand": 0.3,
    "color": 0.2,
    "government": 0.2,
    "center": 0.15,
    "street": 0.15,
}


def translate_text(text, target_lang="en"):
    if not text or len(text) < 3:
        return text
    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except LangDetectException:
        return text
    return text


def calculate_similarity(lost, found):
    total_score = 0
    total_weight = sum(attributes.values())
    for attr, weight in attributes.items():
        text1 = translate_text(str(lost[attr]))
        text2 = translate_text(str(found[attr]))
        embedding1 = text_model.encode(text1, convert_to_tensor=True)
        embedding2 = text_model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        total_score += similarity * weight
    return total_score / total_weight


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()


def build_faiss_index(image_folder):
    embeddings = []
    paths = []
    for img in os.listdir(image_folder):
        if img.startswith("."):
            continue
        path = os.path.join(image_folder, img)
        emb = get_image_embedding(path)
        embeddings.append(emb)
        paths.append(path)
    embedding_matrix = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index, paths


def find_most_similar_images_faiss(input_image_path, image_folder, top_k=3):
    emb = get_image_embedding(input_image_path).reshape(1, -1).astype("float32")
    faiss.normalize_L2(emb)
    index, paths = build_faiss_index(image_folder)
    distances, indices = index.search(emb, top_k)
    results = []
    for i in range(top_k):
        result = {
            "image_path": paths[indices[0][i]],
            "similarity": float(distances[0][i])
        }
        results.append(result)
    return results


@app.post("/match/")
async def match_items(
    match_type: MatchType = Form(...),
    lost_brand: str = Form(""),
    lost_color: str = Form(""),
    lost_government: str = Form(""),
    lost_center: str = Form(""),
    lost_street: str = Form(""),
    lost_contact: str = Form(""),
    found_brand: str = Form(""),
    found_color: str = Form(""),
    found_government: str = Form(""),
    found_center: str = Form(""),
    found_street: str = Form(""),
    found_contact: str = Form(""),
    image: UploadFile = File(None),
):
    # تحقق من الحقول بناءً على نوع المطابقة
    if match_type in ["text", "both"]:
        text_data = MatchTextData(
            match_type=match_type,
            lost_brand=lost_brand,
            lost_color=lost_color,
            lost_government=lost_government,
            lost_center=lost_center,
            lost_street=lost_street,
            lost_contact=lost_contact,
            found_brand=found_brand,
            found_color=found_color,
            found_government=found_government,
            found_center=found_center,
            found_street=found_street,
            found_contact=found_contact,
        )
    if match_type in ["image", "both"] and not image:
        raise HTTPException(status_code=400, detail="Image is required for image or both match types.")

    lost_item = {
        "brand": lost_brand,
        "color": lost_color,
        "government": lost_government,
        "center": lost_center,
        "street": lost_street,
        "contact": lost_contact,
    }

    found_item = {
        "brand": found_brand,
        "color": found_color,
        "government": found_government,
        "center": found_center,
        "street": found_street,
        "contact": found_contact,
    }

    image_score = 0
    text_score = 0
    final_score = 0
    matched_images = []

    try:
        if match_type in ["image", "both"] and image:
            temp_filename = f"{uuid.uuid4().hex}_{image.filename}"
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            image_results = find_most_similar_images_faiss(temp_path, IMAGE_FOLDER, top_k=3)

            matched_images = [
                {
                    "image_url": f"/static/images_db/{os.path.basename(res['image_path'])}",
                    "image_similarity": round(res["similarity"], 4)
                }
                for res in image_results
            ]
            image_score = image_results[0]["similarity"]

        if match_type in ["text", "both"]:
            text_score = calculate_similarity(lost_item, found_item)

        if match_type == "text":
            final_score = text_score
        elif match_type == "image":
            final_score = image_score
        elif match_type == "both":
            final_score = (text_score * 0.5) + (image_score * 0.5)

        matched = 1 if final_score > 0.7 else 0

        return JSONResponse({
            "match_type": match_type,
            "matched_images": matched_images,
            "image_similarity": round(image_score, 4),
            "text_similarity": round(text_score, 4),
            "final_score": round(final_score, 4),
            "matched": matched
        })

    finally:
        if match_type in ["image", "both"] and image:
            os.remove(temp_path)

