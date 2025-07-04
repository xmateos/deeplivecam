import os
import shutil
from typing import Any
import insightface
import json
import pickle
import dill
from types import SimpleNamespace

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, get_temp_source_directory_path, create_temp, create_source_temp, extract_frames, clean_temp, get_temp_frame_paths, get_temp_source_frame_paths
from pathlib import Path

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        #FACE_ANALYSER.prepare(ctx_id=0, det_size=(1920, 1920), det_thresh=0.1)
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None

def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []

        # Cache file
        video_name = os.path.splitext(os.path.basename(modules.globals.target_path))[0]
        duration = get_video_duration_seconds(modules.globals.target_path)
        size = os.path.getsize(modules.globals.target_path)
        temp_directory_path = get_temp_directory_path(modules.globals.target_path)
        cache_file = os.path.join(temp_directory_path, f"0x{video_name}-{duration}s-{size}b.json")
        #cache_file = os.path.join(temp_directory_path, f"0x{video_name}-{duration}s-{size}b.pkl")
        #cache_mapping_file = os.path.join(temp_directory_path, f"0x{video_name}-{duration}s-{size}b.dill")
        #print(f"🧠 Loading cached face data from {cache_file}")
        #print('🧪 No cache found. Creating temp resources...')

        if os.path.exists(cache_file):
            print(f"🧠 Loading cached face data from {cache_file}")
            with open(cache_file, "r") as f:
                loaded = json.load(f)

            new_map = []
            for entry in loaded:
                new_entry = {'id': entry['id'], 'target_faces_in_frame': []}
                for tf in entry['target']:
                    faces = []
                    for d in tf['faces']:
                        face = SimpleNamespace()
                        face.bbox = np.array(d['bbox'])
                        face.kps = np.array(d['kps']) if d.get('kps') is not None else None
                        face.landmark_2d_106 = np.array(d['landmark_2d_106']) if d.get(
                            'landmark_2d_106') is not None else None
                        face.landmark_3d_68 = np.array(d['landmark_3d_68']) if d.get(
                            'landmark_3d_68') is not None else None
                        face.pose = np.array(d['pose']) if d.get('pose') is not None else None
                        face.det_score = d['det_score']
                        face.gender = d.get('gender')
                        face.age = d.get('age')
                        face.target_centroid = d['target_centroid']
                        faces.append(face)
                    new_entry['target_faces_in_frame'].append(
                        {'frame': tf['frame'], 'faces': faces, 'location': tf['location']})
                new_map.append(new_entry)

            modules.globals.source_target_map = new_map
        else:
            print('🧪 No cache found. Creating temp resources...')
            clean_temp(modules.globals.target_path)
            create_temp(modules.globals.target_path)
            create_source_temp(modules.globals.target_path)
            print('Extracting frames...')
            extract_frames(modules.globals.target_path)

            #temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
            temp_frame_paths = get_temp_source_frame_paths(modules.globals.target_path)

            i = 0
            for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
                temp_frame = cv2.imread(temp_frame_path)
                many_faces = get_many_faces(temp_frame)

                for face in many_faces:
                    face_embeddings.append(face.normed_embedding)

                frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': temp_frame_path})
                i += 1

    #        for i, frame in enumerate(frame_face_embeddings):
    #            try:
    #                with open(os.path.join(temp_directory_path, f"0x{video_name}-{duration}s-{size}b_{i}.dill"), "wb") as f:
    #                    dill.dump(frame, f)
    #            except Exception as e:
    #                print(f"❌ Fallo en frame {i}: {type(e)} {e}")
    #                print("Contenido problemático:", frame)
    #                break
    #
    #        try:
    #            with open(cache_file, "wb") as f:
    #                #print("pickle =", pickle)
    #                #print("type(pickle) =", type(pickle))
    #                dill.dump(frame_face_embeddings, f)
    #        except Exception as e:
    #            print("❌ Pickle failed:", type(e), e)

            #centroids = find_cluster_centroids(face_embeddings)
            # Increase detections with new function
            centroids = find_cluster_centroids(face_embeddings, max_k=10, elbow_tolerance=0.05)

            for frame in frame_face_embeddings:
                for face in frame['faces']:
                    closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                    face['target_centroid'] = closest_centroid_index

            for i in range(len(centroids)):
                modules.globals.source_target_map.append({
                    'id' : i
                })

                temp = []
                for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                    temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

                modules.globals.source_target_map[i]['target_faces_in_frame'] = temp

            serializable_map = [
                {
                    'id': entry['id'],
                    'target': [
                        {
                            'frame': tf['frame'],
                            'location': tf['location'],
                            'faces': [
                                {
                                    'bbox': face.bbox.tolist(),
                                    'kps': face.kps.tolist(),
                                    'landmark_2d_106': face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None,
                                    'landmark_3d_68': face.landmark_3d_68.tolist() if hasattr(face, 'landmark_3d_68') else None,
                                    'pose': face.pose.tolist() if hasattr(face, 'pose') else None,
                                    'det_score': float(face.det_score),
                                    'gender': int(face.gender) if hasattr(face, 'gender') else None,
                                    'age': int(face.age) if hasattr(face, 'age') else None,
                                    'target_centroid': int(face.target_centroid)
                                }
                                for face in tf['faces']
                            ]
                        } for tf in entry['target_faces_in_frame']
                    ]
                }
                for entry in modules.globals.source_target_map
            ]
            with open(cache_file, "w") as f:
                json.dump(serializable_map, f, indent=2)

            #dump_faces(centroids, frame_face_embeddings)

        default_target_face()
    except ValueError:
        return None
    

def get_video_duration_seconds(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(frame_count / fps) if fps > 0 else 0


def get_attr(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def default_target_face():
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if get_attr(face, 'det_score') > get_attr(best_face, 'det_score'):
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = get_attr(best_face, 'bbox')

        target_frame = cv2.imread(best_frame['location'])
        map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
                        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1