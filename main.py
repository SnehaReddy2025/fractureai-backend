import os
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io, base64, threading, time, urllib.request
from pathlib import Path

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse({}, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "*"
        })
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

print("Loading model...")
model = YOLO("best_v2.pt")
print("Model loaded!")

def keep_alive():
    time.sleep(60)
    while True:
        try:
            url = os.environ.get("RENDER_EXTERNAL_URL", "")
            if url:
                urllib.request.urlopen(url + "/health", timeout=10)
        except: pass
        time.sleep(600)
threading.Thread(target=keep_alive, daemon=True).start()

@app.get("/")
def root(): return {"status": "ok", "app": "FractureAI"}

@app.get("/health")
def health(): return {"status": "ok", "model": "YOLOv8s-v2", "accuracy": "97.2%"}

@app.post("/detect")
async def detect(file: UploadFile = File(...), body_part: str = Form(default="auto")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model.predict(source=image, conf=0.45, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({"box":[int(x1),int(y1),int(x2),int(y2)],
                    "confidence":round(conf,3),"label":"fracture"})
        draw = ImageDraw.Draw(image)
        for det in detections:
            x1,y1,x2,y2 = det["box"]
            draw.rectangle([x1,y1,x2,y2], outline="#ef4444", width=4)
            label = "Fracture " + str(round(det["confidence"]*100)) + "%"
            draw.rectangle([x1,y1-22,x1+len(label)*8,y1], fill="#ef4444")
            draw.text((x1+4,y1-20), label, fill="white")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse(content={
            "success": True,
            "fractures_found": len(detections),
            "fracture_detected": len(detections) > 0,
            "summary": str(len(detections)) + " fracture(s) detected." if detections else "No fractures detected.",
            "severity": "HIGH" if any(d["confidence"]>0.9 for d in detections) else "MODERATE" if detections else "NONE",
            "findings": ["Fracture detected (" + str(round(d["confidence"]*100)) + "% confidence)" for d in detections] or ["No fracture identified."],
            "recommendations": ["Orthopaedic consultation advised.", "AI-assisted — confirm with radiologist."] if detections else ["No fracture detected."],
            "annotated_image": "data:image/png;base64," + img_b64,
            "detections": detections
        }, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)},
            status_code=500, headers={"Access-Control-Allow-Origin": "*"})
