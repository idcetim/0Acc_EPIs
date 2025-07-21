import cv2
import numpy as np
import yaml
import json
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------- CONFIGURACIÓN ----------
MODEL_PATH = 'best.pt'
DATA_YAML_PATH = 'data.yaml'
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ID_MAP_PATH = 'ids.json'
INPUT_VIDEO = 'D:/Datasets/0Accidentes/Fatigue/20250718_test_lat_derecha_fatiga1 - Nada_Muy bajo.mp4'
OUTPUT_VIDEO = 'output_video_con_2epis.mp4'
OUTPUT_JSON = 'alertas2.json'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
ALPHA = 0.4
WINDOW_SECONDS = 3

# ---------- FUNCIONES DE UTILIDAD ----------

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data['names']
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names

def load_id_map(json_path):
    with open(json_path, 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}

def get_color(label):
    alert_colors = {
        'no helmet': (0, 0, 255),
        'no nest': (0, 69, 255),
        'no gloves': (0, 0, 180),
        'no boots': (0, 140, 255),
    }
    safe_colors = {
        'helmet': (0, 255, 0),
        'vest': (0, 255, 255),
        'gloves': (255, 255, 0),
        'boots': (255, 0, 0),
    }
    return alert_colors.get(label, safe_colors.get(label, (200, 200, 200)))

def draw_transparent_box(img, box, color, alpha=0.4):
    overlay = img.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def is_inside(inner_box, outer_box, margin=0.1):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    margin_x = (ox2 - ox1) * margin
    margin_y = (oy2 - oy1) * margin
    return (
        ix1 >= ox1 - margin_x and iy1 >= oy1 - margin_y and
        ix2 <= ox2 + margin_x and iy2 <= oy2 + margin_y
    )

def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def find_nearest_worker(aruco_center, worker_boxes):
    min_dist = float('inf')
    best_id = None
    for wid, box in worker_boxes.items():
        cx, cy = get_centroid(box)
        dist = np.hypot(cx - aruco_center[0], cy - aruco_center[1])
        if dist < min_dist:
            min_dist = dist
            best_id = wid
    return best_id if min_dist < 150 else None  # threshold en píxeles

# ---------- PROCESAMIENTO PRINCIPAL ----------

def process_video_with_tracking():
    class_names = load_class_names(DATA_YAML_PATH)
    id_map = load_id_map(ID_MAP_PATH)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    window_size = int(WINDOW_SECONDS * fps)

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    tracker = DeepSort(max_age=30)

    frame_idx = 0
    alert_data = []
    epi_presence = defaultdict(lambda: defaultdict(list))
    aruco_to_track = {}       # ArucoID -> track_id
    track_to_aruco = {}       # track_id -> ArucoID

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        humans = []
        epis = []

        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].item())
            label = class_names[cls_id].strip().lower()
            if label == 'human':
                humans.append((xyxy, box.conf[0].item()))
            else:
                epis.append((xyxy, label))

        # Tracking
        detections = []
        for bbox, conf in humans:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'human'))

        tracks = tracker.update_tracks(detections, frame=frame)
        worker_boxes = {}

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = f"worker_{track.track_id}"
            l, t, w, h = track.to_ltrb()
            box = [int(l), int(t), int(w), int(h)]
            worker_boxes[tid] = box

        # Detectar ArUcos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            for i, corner in enumerate(corners):
                aruco_id = int(ids[i][0])
                c = corner[0].mean(axis=0).astype(int)
                cx, cy = int(c[0]), int(c[1])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"ID: {aruco_id}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                nearest_tid = find_nearest_worker((cx, cy), worker_boxes)
                if nearest_tid:
                    aruco_to_track[aruco_id] = nearest_tid
                    track_to_aruco[nearest_tid] = aruco_id

        # Dibujar boxes y EPIs
        for tid, box in worker_boxes.items():
            name = tid
            if tid in track_to_aruco:
                aruco_id = track_to_aruco[tid]
                if aruco_id in id_map:
                    name = f"{id_map[aruco_id]} (ID: {aruco_id})"
                else:
                    name = f"ID: {aruco_id}"
            frame = draw_transparent_box(frame, box, (255, 255, 255), alpha=ALPHA)
            cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Asociar EPIs a trabajadores
        for tid, person_box in worker_boxes.items():
            worker_epis = set()
            for epi_box, epi_label in epis:
                if is_inside(epi_box, person_box):
                    worker_epis.add(epi_label)
            for epi in ['helmet', 'vest', 'gloves', 'boots']:
                epi_presence[tid][epi].append(epi in worker_epis)

        for epi_box, epi_label in epis:
            for person_box in worker_boxes.values():
                if is_inside(epi_box, person_box):
                    color = get_color(epi_label)
                    frame = draw_transparent_box(frame, epi_box, color, alpha=ALPHA)
                    x1, y1 = int(epi_box[0]), int(epi_box[1])
                    cv2.putText(frame, f"{epi_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break

        out.write(frame)

        # JSON
        if (frame_idx + 1) % window_size == 0:
            segment = {
                "time_window": f"{int(frame_idx/fps - WINDOW_SECONDS):02d}s - {int(frame_idx/fps):02d}s",
                "workers": []
            }
            for tid, epis_dict in epi_presence.items():
                summary = {
                    "id_worker": tid,
                    "epis_present": [],
                    "epis_missing": []
                }
                if tid in track_to_aruco:
                    aruco_id = track_to_aruco[tid]
                    if aruco_id in id_map:
                        summary["id_worker"] = id_map[aruco_id]
                for epi, values in epis_dict.items():
                    presence_rate = sum(values) / len(values)
                    if presence_rate >= 0.5:
                        summary["epis_present"].append(epi)
                    else:
                        summary["epis_missing"].append(epi)
                segment["workers"].append(summary)
            alert_data.append(segment)
            epi_presence.clear()

        frame_idx += 1

    cap.release()
    out.release()

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(alert_data, f, indent=2)

    print(f"✅ Video generado: {OUTPUT_VIDEO}")
    print(f"📄 JSON de alertas generado: {OUTPUT_JSON}")

if __name__ == '__main__':
    process_video_with_tracking()