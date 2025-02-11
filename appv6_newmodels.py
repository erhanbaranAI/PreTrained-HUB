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
import matplotlib.pyplot as plt
import io
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 📌 Model indirme linkleri
MODEL_LINKS = {
    # YOLO Serisi
    #  YOLOv5n: https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n-cls.pt
    "YOLOv5s": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s-cls.pt",
    "YOLOv5m": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m-cls.pt",
    "YOLOv5l": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l-cls.pt",
    "YOLOv5x": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x-cls.pt",
    
    "YOLOv7": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
    "YOLOv8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "YOLOv8m": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "YOLOv8l": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
    "YOLOv8x": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",

    # Torchvision Detection Modelleri
    "Faster R-CNN": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "SSD": "https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth",
    "RetinaNet": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
    "Mask R-CNN": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
    "Keypoint R-CNN": "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
    "DETR": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",

    # Segmentasyon Modelleri
    "DeepLabV3": "https://download.pytorch.org/models/deeplabv3_resnet50_coco-586e9d54.pth"
}

# 📌 Model türlerini belirleme
MODEL_TYPES = {name: "yolo" if "YOLO" in name else "torchvision" for name in MODEL_LINKS.keys()}

# 📌 Model yolları
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

# 📌 Modelleri indir ve yükle
download_models()
models_dict = {}

for name, path in MODEL_PATHS.items():
    try:
        if MODEL_TYPES[name] == "yolo":
            models_dict[name] = YOLO(path)
        else:
            if name == "Faster R-CNN":
                models_dict[name] = models.detection.fasterrcnn_resnet50_fpn(weights=None)
            elif name == "SSD":
                models_dict[name] = models.detection.ssd300_vgg16(weights=None)
            elif name == "RetinaNet":
                models_dict[name] = models.detection.retinanet_resnet50_fpn(weights=None)
            elif name == "Mask R-CNN":
                models_dict[name] = models.detection.maskrcnn_resnet50_fpn(weights=None)
            elif name == "Keypoint R-CNN":
                models_dict[name] = models.detection.keypointrcnn_resnet50_fpn(weights=None)
            elif name == "DETR":
                models_dict[name] = models.detection.detr_resnet50(weights=None)
            elif name == "DeepLabV3":
                models_dict[name] = models.segmentation.deeplabv3_resnet50(weights=None)

            models_dict[name].load_state_dict(torch.load(path))
            models_dict[name].eval()
        
        print(f"✅ {name} başarıyla yüklendi.")

    except Exception as e:
        print(f"❌ {name} yüklenirken hata oluştu: {e}")

def detect_objects(image, selected_models, threshold=0.5, thickness=2, color="#00FF00"):
    """
    Seçili modeller ile görüntüde nesne tespiti yapar.
    Kullanıcı tarafından belirlenen threshold, çizgi kalınlığı ve renk ayarlarını uygular.
    """
    
    # 📌 Varsayılan çıktı yapısı (Tüm modeller için)
    output_results = {model_name: (None, "", "") for model_name in MODEL_PATHS.keys()}
    
    inference_times = {}  # Inference sürelerini saklar
    detection_counts = {}  # Tespit edilen nesne sayısını saklar

    if not selected_models:
        return tuple(output_results.values())  # Hata almamak için statik çıktı

    # 🎨 Renk kodunun doğru formatta olduğunu kontrol et
    try:
        color = color.lstrip("#")
        color = (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))  # RGB formatı
        color = color[::-1]  # OpenCV için BGR formatı
    except Exception as e:
        print(f"🚨 Renk dönüştürme hatası: {e}")
        color = (0, 255, 0)  # Varsayılan yeşil BGR

    for model_name in selected_models:
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

                # Eğer model bounding box içeriyorsa (Detections)
                if "boxes" in predictions:
                    boxes = predictions["boxes"].detach().cpu().numpy()
                    labels = predictions["labels"].detach().cpu().numpy()
                    scores = predictions["scores"].detach().cpu().numpy()
                    
                    filtered_indices = scores >= threshold  # Threshold filtresi
                    boxes, labels, scores = boxes[filtered_indices], labels[filtered_indices], scores[filtered_indices]

                    detected_objects = [(int(label), round(float(score), 2)) for label, score in zip(labels, scores)]
                    output_image = np.array(image)
                    output_image = draw_boxes(output_image, boxes, labels, scores, thickness, color)

                # Eğer model segmentasyon yapıyorsa (örneğin Mask R-CNN, DeepLabV3)
                elif "masks" in predictions:
                    masks = predictions["masks"].detach().cpu().numpy()
                    masks = masks > 0.5  # Maske eşik değeri uygula

                    output_image = np.array(image)
                    for mask in masks:
                        mask = (mask * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(output_image, contours, -1, color, thickness)

                    detected_objects = [("Segmentation", len(masks))]

                else:
                    detected_objects = [("No objects detected", 0)]
                    output_image = np.array(image)

            end_time = time.time()
            inference_time = round(end_time - start_time, 3)

            print(f"✅ {model_name} ÇALIŞTI! - {inference_time} saniye")
            inference_times[model_name] = inference_time  # Inference süresini kaydet
            detection_counts[model_name] = len(detected_objects)  # Tespit edilen nesne sayısını kaydet
            output_results[model_name] = (output_image, str(detected_objects), f"{model_name} tamamlandı - {inference_time} saniye")

        except Exception as e:
            print(f"🚨 HATA! {model_name} çalışırken hata oluştu: {e}")

    # 📌 Grafik Çıktısı (Performans Analizi)
    graph_figure = plot_results(inference_times, detection_counts)
    
    return (
        *[output_results[model][0:3] for model in MODEL_PATHS.keys()],
        graph_figure  # Base64 yerine Matplotlib figürünü döndür
    )

def draw_boxes(image, boxes, labels, scores, thickness=2, color=(0, 255, 0)):
    """Tespit edilen nesneleri görüntü üzerine çizer."""
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def plot_results(inference_times, detection_counts):
    """
    Inference süresi ve tespit edilen nesne sayısını karşılaştıran bir grafik oluşturur.
    """
    models = list(inference_times.keys())
    times = list(inference_times.values())
    detections = list(detection_counts.values())

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Çubuk grafiği (Inference Süresi)
    ax1.bar(models, times, color='b', alpha=0.6, label="Inference Süresi (saniye)")

    # İkinci y ekseni (Tespit Edilen Nesne Sayısı)
    ax2 = ax1.twinx()
    ax2.plot(models, detections, color='r', marker='o', markersize=8, linewidth=2, label="Tespit Edilen Nesne Sayısı")

    # Eksen etiketleri
    ax1.set_xlabel("Modeller")
    ax1.set_ylabel("Inference Süresi (saniye)", color='b')
    ax2.set_ylabel("Tespit Edilen Nesne Sayısı", color='r')

    # Başlık ve Grid
    ax1.set_title("Modellerin Performans Karşılaştırması")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    return fig  # Base64 yerine doğrudan fig nesnesini döndür


# 📌 Arayüz
with gr.Blocks() as demo:
    gr.Markdown("# MLModelHub - Önceden Eğitilmiş Modellerle Nesne Tespiti")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Resim Yükle")
    with gr.Row():
        model_selection = gr.CheckboxGroup(choices=list(MODEL_PATHS.keys()), label="Modelleri Seç")

    detect_button = gr.Button("Nesneleri Bul")
    output_box = gr.Textbox(label="Sonuçlar")

    detect_button.click(
        detect_objects,
        inputs=[image_input, model_selection],
        outputs=[output_box]
    )

demo.launch()


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
