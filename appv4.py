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

def draw_boxes(image, boxes, labels, scores, thickness=2, color=(0, 255, 0)):
    """Tespit edilen nesneleri görüntü üzerine çizer."""
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def detect_objects(image, selected_models, threshold=0.5, thickness=2, color="#00FF00"):
    """
    Seçili modeller ile görüntüde nesne tespiti yapar.
    Kullanıcı tarafından belirlenen threshold, çizgi kalınlığı ve renk ayarlarını uygular.
    """
    output_results = {
        "YOLOv5s": (None, "", ""),
        "YOLOv8n": (None, "", ""),
        "Faster R-CNN": (None, "", ""),
        "SSD": (None, "", "")
    }

    if not selected_models:
        return tuple(output_results.values())  # Hata almamak için statik çıktı

    # 🎨 Renk kodunun doğru formatta olduğunu kontrol et
    if not isinstance(color, str) or not color.startswith("#"):
        color = "#00FF00"  # Varsayılan yeşil

    try:
        # 📌 HEX -> RGB çevir
        color = color.lstrip("#")
        color = (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))  # RGB formatı
        color = color[::-1]  # OpenCV için BGR formatı
    except Exception as e:
        print(f"🚨 Renk dönüştürme hatası: {e}")
        color = (0, 255, 0)  # Varsayılan yeşil BGR

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
                    results = [r for r in results if r.boxes.conf.max() >= threshold]  # Threshold filtresi
                    output_image = results[0].plot(line_width=thickness)
                    detected_objects = [(model.names[int(box.cls[0])], round(float(box.conf[0]), 2)) for result in results for box in result.boxes]

                elif model_type == "torchvision":
                    transform = transforms.Compose([transforms.ToTensor()])
                    image_tensor = transform(image).unsqueeze(0)
                    predictions = model(image_tensor)[0]

                    boxes = predictions["boxes"].detach().cpu().numpy()
                    labels = predictions["labels"].detach().cpu().numpy()
                    scores = predictions["scores"].detach().cpu().numpy()

                    filtered_indices = scores >= threshold  # Threshold filtresi
                    boxes, labels, scores = boxes[filtered_indices], labels[filtered_indices], scores[filtered_indices]

                    detected_objects = [(int(label), round(float(score), 2)) for label, score in zip(labels, scores)]
                    output_image = np.array(image)
                    output_image = draw_boxes(output_image, boxes, labels, scores, thickness, color)

                end_time = time.time()
                inference_time = round(end_time - start_time, 3)

                print(f"✅ {model_name} ÇALIŞTI! - {inference_time} saniye")

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

# 📌 Gradio Arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# MLModelHub - Önceden Eğitilmiş Modellerle Nesne Tespiti")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="🖼 Resim Yükle")

    with gr.Row():
        model_selection = gr.CheckboxGroup(choices=list(MODEL_PATHS.keys()), label="📌 Modelleri Seç")

    with gr.Accordion("⚙️ **Gelişmiş Ayarlar**", open=False):
        threshold_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="🎯 Threshold")
        thickness_slider = gr.Slider(1, 10, value=2, step=1, label="🖍 Çizgi Kalınlığı")
        color_picker = gr.ColorPicker(value="#00FF00", label="🎨 Kutu Rengi")

    detect_button = gr.Button("🚀 Nesneleri Bul")

    # 📌 Her model için sabit çıktı alanları
    yolo_v5_image = gr.Image(label="YOLOv5s - Sonuç")
    yolo_v5_objects = gr.Textbox(label="YOLOv5s - Tespit Edilen Nesneler")
    yolo_v5_time = gr.Textbox(label="YOLOv5s - Inference Süresi")

    yolo_v8_image = gr.Image(label="YOLOv8n - Sonuç")
    yolo_v8_objects = gr.Textbox(label="YOLOv8n - Tespit Edilen Nesneler")
    yolo_v8_time = gr.Textbox(label="YOLOv8n - Inference Süresi")

    faster_rcnn_image = gr.Image(label="Faster R-CNN - Sonuç")
    faster_rcnn_objects = gr.Textbox(label="Faster R-CNN - Tespit Edilen Nesneler")
    faster_rcnn_time = gr.Textbox(label="Faster R-CNN - Inference Süresi")

    ssd_image = gr.Image(label="SSD - Sonuç")
    ssd_objects = gr.Textbox(label="SSD - Tespit Edilen Nesneler")
    ssd_time = gr.Textbox(label="SSD - Inference Süresi")

    detect_button.click(
        detect_objects, 
        inputs=[image_input, model_selection, threshold_slider, thickness_slider, color_picker], 
        outputs=[
            yolo_v5_image, yolo_v5_objects, yolo_v5_time,
            yolo_v8_image, yolo_v8_objects, yolo_v8_time,
            faster_rcnn_image, faster_rcnn_objects, faster_rcnn_time,
            ssd_image, ssd_objects, ssd_time
        ]
    )

# 📌 Yeni Sayfa (Multi-page Route)
with demo.route("YOLOv5s", "/yolo-v5s"):
        gr.Markdown("# 🚀 YOLOv5s - Derinlemesine İnceleme")
        
        gr.Markdown("### **📌 YOLOv5s Nedir?**")
        gr.Markdown(
            "YOLOv5s (You Only Look Once v5 - Small), **Ultralytics** tarafından geliştirilen "
            "**YOLO nesne tespiti** ailesine ait bir modeldir. 's' versiyonu, **hafif, hızlı ve düşük gecikmeli** "
            "olmasıyla bilinir. Küçük boyutu sayesinde **mobil cihazlar ve gömülü sistemlerde kolayca çalışabilir**. "
            "**COCO dataset** üzerinde eğitilmiş olup 80 farklı nesneyi tanıyabilir."
        )

        gr.Markdown("### **🛠 YOLOv5s'nin Mimari Yapısı**")
        gr.Markdown(
            "**YOLOv5s modeli üç ana bileşenden oluşur:**\n\n"
            "**1️⃣ Backbone (Öznitelik Çıkarımı):** CSPDarknet53 kullanır, residual bloklarla optimize edilmiştir.\n\n"
            "**2️⃣ Neck (Özellik Birleştirme):** PAFPN (Path Aggregation Feature Pyramid Network) ile FPN+PAN yapısı kullanır.\n\n"
            "**3️⃣ Head (Tahmin Aşaması):** Nesne kutularını tahmin eder, GIoU Loss ile optimize edilmiştir.\n"
        )

        gr.Markdown("#### **1️⃣ Backbone - Feature Extractor**")
        gr.Textbox(
            "Backbone, modelin ham görüntüden özellikler çıkardığı kısımdır. "
            "CSPDarknet53 kullanılarak residual bağlantılar eklenmiş ve hesaplama verimliliği artırılmıştır. "
            "Bu bölüm, çeşitli konvolüsyon katmanları ve aktivasyon fonksiyonları içerir.",
            label="📌 Backbone Açıklaması"
        )

        gr.Markdown("#### **2️⃣ Neck - Özellik Birleştirme**")
        gr.Textbox(
            "Neck katmanı, farklı ölçeklerdeki nesneleri tespit edebilmek için geliştirilmiştir. "
            "PAFPN (Path Aggregation Feature Pyramid Network) ile FPN ve PAN bileşenlerini kullanarak "
            "küçük ve büyük nesnelerin daha iyi algılanmasını sağlar.",
            label="📌 Neck Açıklaması"
        )

        gr.Markdown("#### **3️⃣ Head - Tahmin Aşaması**")
        gr.Textbox(
            "Head bölümü, nesne tespiti için nihai tahminleri üretir. Modelin bounding box koordinatlarını, "
            "nesne puanlarını ve sınıf tahminlerini içerir. Sigmoid aktivasyon fonksiyonu ile olasılıklar hesaplanır.",
            label="📌 Head Açıklaması"
        )

        gr.Markdown("### **📌 Önceden Eğitilmiş Model (Pretrained) İçerisindeki Sınıflar**")
        gr.Textbox(
            "YOLOv5s, COCO veri seti üzerinde eğitilmiştir ve 80 farklı sınıfı tespit edebilir:\n"
            "Person, Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck, Boat, Traffic light...\n"
            "(Tüm sınıf listesi içeriğe eklenmiştir)",
            label="📌 Sınıf Listesi"
        )

        gr.Markdown("### **📌 YOLOv5s Kullanım Alanları**")
        gr.Textbox(
            "📷 **Güvenlik Kameraları**: Gerçek zamanlı izleme için kullanılır.\n"
            "🚗 **Otonom Araçlar**: Yol üzerindeki nesneleri tanır.\n"
            "🏭 **Endüstriyel Üretim**: Ürün kusurlarını tespit eder.\n"
            "🧑‍⚕️ **Tıbbi Görüntüleme**: Radyolojik analizlerde kullanılabilir.\n"
            "🌿 **Tarım**: Bitki hastalıklarını ve böcekleri belirleyebilir.\n"
            "🛒 **Perakende**: Raf düzeni ve envanter yönetimi için kullanılır.",
            label="📌 Kullanım Alanları"
        )

with demo.route("YOLOv8n", "/yolo-v8n"):
    gr.Markdown("# 📂 Yeni Sayfa - Ekstra Bilgiler")
    gr.Markdown("Burada ek bilgileri gösterebilirsiniz!")
    gr.Textbox("Bu, yeni sayfada gösterilecek bir içeriktir.", label="📌 Açıklama")

with demo.route("Faster R-CNN", "/faster-r-cnn"):
    gr.Markdown("# 📂 Yeni Sayfa - Ekstra Bilgiler")
    gr.Markdown("Burada ek bilgileri gösterebilirsiniz!")
    gr.Textbox("Bu, yeni sayfada gösterilecek bir içeriktir.", label="📌 Açıklama")

with demo.route("SSD", "/ssd"):
    gr.Markdown("# 📂 Yeni Sayfa - Ekstra Bilgiler")
    gr.Markdown("Burada ek bilgileri gösterebilirsiniz!")
    gr.Textbox("Bu, yeni sayfada gösterilecek bir içeriktir.", label="📌 Açıklama")


demo.launch()
