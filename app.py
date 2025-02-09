import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
import os
import urllib.request
from ultralytics import YOLO

# 📌 Model indirme linkleri
MODEL_LINKS = {
    "YOLOv5s": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5su.pt",
    "YOLOv8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "Faster R-CNN": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "SSD": "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth"
}

MODEL_TYPES = {
    "YOLOv5s": "yolo",
    "YOLOv8n": "yolo",
    "Faster R-CNN": "torchvision",
    "SSD": "torchvision"
}

MODEL_PATHS = {name: f"models/{name.replace(' ', '_').lower()}.pt" for name in MODEL_LINKS.keys()}

# 📌 Modelleri indir
def download_models():
    os.makedirs("models", exist_ok=True)
    for model_name, url in MODEL_LINKS.items():
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            print(f"📥 {model_name} indiriliyor...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ {model_name} başarıyla indirildi!")
        else:
            print(f"📁 {model_name} zaten mevcut, indirme atlandı.")

# 📌 Model indir ve yükle
download_models()
models_dict = {}

for name, path in MODEL_PATHS.items():
    try:
        model_type = MODEL_TYPES[name]

        if model_type == "yolo":
            models_dict[name] = YOLO(path)

        elif model_type == "torchvision":
            if name == "Faster R-CNN":
                models_dict[name] = models.detection.fasterrcnn_resnet50_fpn(weights=None)
                models_dict[name].load_state_dict(torch.load(path))
                models_dict[name].eval()

            elif name == "SSD":
                models_dict[name] = models.detection.ssd300_vgg16(weights=None)
                models_dict[name].load_state_dict(torch.load(path))
                models_dict[name].eval()

        print(f"✅ {name} başarıyla yüklendi.")

    except Exception as e:
        print(f"❌ {name} yüklenirken hata oluştu: {e}")

def draw_boxes(image, boxes, labels, scores):
    """Tespit edilen nesneleri görüntü üzerine çizer."""
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def detect_objects(image, selected_models):
    """
    Seçili modeller ile görüntüde nesne tespiti yapar.
    Tüm modeller için çıktı döndürerek Gradio'nun hata vermesini engeller.
    """
    output_results = {
        "YOLOv5s": (None, "", ""),
        "YOLOv8n": (None, "", ""),
        "Faster R-CNN": (None, "", ""),
        "SSD": (None, "", "")
    }

    if not selected_models:
        return tuple(output_results.values())  # Hata almamak için statik çıktı

    for model_name in output_results.keys():
        if model_name in selected_models:
            model = models_dict.get(model_name, None)
            if not model:
                print(f"🚨 {model_name} modeli YÜKLENEMEDİ!")
                continue

            model_type = MODEL_TYPES[model_name]

            # 📌 Model inference başlat
            start_time = time.time()

            try:
                if model_type == "yolo":
                    results = model(image)
                    output_image = results[0].plot()
                    detected_objects = [(model.names[int(box.cls[0])], round(float(box.conf[0]), 2)) for result in results for box in result.boxes]

                elif model_type == "torchvision":
                    transform = transforms.Compose([transforms.ToTensor()])
                    image_tensor = transform(image).unsqueeze(0)
                    predictions = model(image_tensor)[0]

                    boxes = predictions["boxes"].detach().cpu().numpy()
                    labels = predictions["labels"].detach().cpu().numpy()
                    scores = predictions["scores"].detach().cpu().numpy()

                    detected_objects = [(int(label), round(float(score), 2)) for label, score in zip(labels, scores)]

                    output_image = np.array(image)
                    output_image = draw_boxes(output_image, boxes, labels, scores)

                end_time = time.time()
                inference_time = round(end_time - start_time, 3)

                # 📌 Debugging için terminale yazdır
                print(f"✅ {model_name} ÇALIŞTI!")
                print(f"📷 Görsel: {type(output_image)}")
                print(f"📌 Tespit Edilen Nesneler: {detected_objects}")
                print(f"⏳ Inference Süresi: {inference_time} saniye")

                # 📌 Sonucu ilgili modelin satırına ekle
                output_results[model_name] = (output_image, str(detected_objects), f"{model_name} tamamlandı - {inference_time} saniye")

            except Exception as e:
                print(f"🚨 HATA! {model_name} çalışırken hata oluştu: {e}")
    
    # 📌 Tüm modellerin çıktısını döndürerek Gradio'nun hata vermesini önle
    return (
        output_results["YOLOv5s"][0], output_results["YOLOv5s"][1], output_results["YOLOv5s"][2],
        output_results["YOLOv8n"][0], output_results["YOLOv8n"][1], output_results["YOLOv8n"][2],
        output_results["Faster R-CNN"][0], output_results["Faster R-CNN"][1], output_results["Faster R-CNN"][2],
        output_results["SSD"][0], output_results["SSD"][1], output_results["SSD"][2]
    )


def select_all():
    """Tüm modelleri seçmek için kullanılan fonksiyon"""
    return list(MODEL_PATHS.keys())

# 📌 Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# MLModelHub - Önceden Eğitilmiş Modellerle Nesne Tespiti")
    gr.Markdown("📌 Bir resim yükleyin ve farklı modellerin nasıl çalıştığını test edin.")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Resim Yükle")
        model_selection = gr.CheckboxGroup(choices=list(MODEL_PATHS.keys()), label="Modelleri Seç")
    
    select_all_button = gr.Button("Tümünü Seç")
    detect_button = gr.Button("Nesneleri Bul")

    output_blocks = {}

    for model_name in MODEL_TYPES.keys():
        with gr.Row():
            output_blocks[model_name] = (
                gr.Image(label=f"{model_name} - Tespit Edilen Nesneler"),
                gr.Textbox(label=f"{model_name} - Nesneler ve Doğruluk Oranları"),
                gr.Textbox(label=f"{model_name} - Inference Süresi")
            )

    detect_button.click(
        detect_objects, 
        inputs=[image_input, model_selection], 
        outputs=[item for block in output_blocks.values() for item in block]
    )
    
    select_all_button.click(select_all, outputs=model_selection)

# 📌 Uygulamayı başlat
if __name__ == "__main__":
    demo.queue()  # 🔹 Asenkron çalıştırmayı sağlıyor
    demo.launch()
