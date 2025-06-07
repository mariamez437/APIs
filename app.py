from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from deepface import DeepFace
from PIL import Image
from enum import Enum
import shutil
import os
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

text_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
yolo_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(yolo_model_path)

attribute_weights = {
    "name": 0.5,
    "national_id": 0.3,
    "governorate": 0.1,
    "city": 0.05,
    "street": 0.05
}

THRESHOLD = 0.75

class MatchType(str, Enum):
    text = "text"
    Image = "image"
    both = "both"

def translate_text(text, target_lang="en"):
    try:
        if not text or len(text) < 3:
            return text
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except LangDetectException:
        return text
    return text

def calculate_text_similarity(lost: dict, found: dict):
    total_score = 0
    total_weight = sum(attribute_weights.values())
    for attr, weight in attribute_weights.items():
        text1 = translate_text(str(lost[attr]))
        text2 = translate_text(str(found[attr]))
        emb1 = text_model.encode(text1, convert_to_tensor=True)
        emb2 = text_model.encode(text2, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        total_score += sim * weight
    return total_score / total_weight

def detect_and_crop_face(image_path, prefix="face"):
    img = Image.open(image_path).convert("RGB")
    result = face_model(img, classes=[0])
    boxes = result[0].boxes.xyxy.cpu().numpy()
    scores = result[0].boxes.conf.cpu().numpy()
    for i, score in enumerate(scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, boxes[i])
            face = img.crop((x1, y1, x2, y2))
            output_path = f"static/faces/{prefix}_face.jpg"
            os.makedirs("static/faces", exist_ok=True)
            face.save(output_path)
            return output_path
    return None

@app.post("/match/")
async def match(
    match_type: MatchType = Form(...),

    lost_name: str = Form(None),
    lost_national_id: str = Form(None),
    lost_governorate: str = Form(None),
    lost_city: str = Form(None),
    lost_street: str = Form(None),

    found_name: str = Form(None),
    found_national_id: str = Form(None),
    found_governorate: str = Form(None),
    found_city: str = Form(None),
    found_street: str = Form(None),

    lost_image: UploadFile = File(None),
    found_image: UploadFile = File(None),
):
    lost = {}
    found = {}
    text_score = None
    face_verified = None
    face_distance = None
    lost_face_url = None
    found_face_url = None

    # Load metadata/contact info from staticdb/metadata.json
    metadata_path = "staticdb/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            contact_info_dict = json.load(f)
    else:
        contact_info_dict = {}

    if match_type in ["text", "both"]:
        required_fields = [lost_name, lost_national_id, lost_governorate, lost_city, lost_street,
                           found_name, found_national_id, found_governorate, found_city, found_street]
        if any(field is None for field in required_fields):
            return {"error": "Missing required text fields"}

        lost = {
            "name": lost_name,
            "national_id": lost_national_id,
            "governorate": lost_governorate,
            "city": lost_city,
            "street": lost_street,
        }

        found = {
            "name": found_name,
            "national_id": found_national_id,
            "governorate": found_governorate,
            "city": found_city,
            "street": found_street,
        }

        text_score = calculate_text_similarity(lost, found)

    if match_type in ["image", "both"]:
        if lost_image is None or found_image is None:
            return {"error": "Missing image files"}

        lost_img_path = f"static/{lost_image.filename}"
        found_img_path = f"static/{found_image.filename}"
        with open(lost_img_path, "wb") as f:
            shutil.copyfileobj(lost_image.file, f)
        with open(found_img_path, "wb") as f:
            shutil.copyfileobj(found_image.file, f)

        try:
            cropped1 = detect_and_crop_face(lost_img_path, prefix="lost")
            cropped2 = detect_and_crop_face(found_img_path, prefix="found")

            if cropped1 and cropped2:
                result = DeepFace.verify(
                    img1_path=cropped1,
                    img2_path=cropped2,
                    model_name="Facenet512", 
                    distance_metric="euclidean_l2",
                    threshold=0.7
                )
                face_verified = result["verified"]
                face_distance = result["distance"]
                lost_face_url = f"/static/faces/lost_face.jpg"
                found_face_url = f"/static/faces/found_face.jpg"
        except Exception as e:
            print(f"Face matching error: {e}")
        finally:
            if os.path.exists(lost_img_path): os.remove(lost_img_path)
            if os.path.exists(found_img_path): os.remove(found_img_path)

    final_result = None
    if match_type == "text":
        final_result = text_score > THRESHOLD
    elif match_type == "image":
        final_result = face_verified
    elif match_type == "both":
        final_result = (text_score and text_score > THRESHOLD) and face_verified

    # Retrieve contact info
    lost_contact = contact_info_dict.get("lost_face.jpg", None)
    found_contact = contact_info_dict.get("found_face.jpg", None)

    return {
        "text_similarity": round(text_score, 4) if text_score is not None else None,
        "face_verified": face_verified,
        "face_distance": face_distance,
        "match_result": final_result,
        "face_images": {
            "lost_face": f"http://localhost:2000{lost_face_url}" if lost_face_url else None,
            "found_face": f"http://localhost:2000{found_face_url}" if found_face_url else None
        },
        "contact_info": {
            
            "found": found_contact
        }
    }

