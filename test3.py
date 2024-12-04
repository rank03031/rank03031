import math
import os
import gc
import cv2
import json
import time
import shutil
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor


# 모델 로드
model = tf.keras.models.load_model('12-14.h5') 

video_path = 'C:/Users/COX/Desktop/영상/00341.mp4' # 입력할 비디오, GUI 인풋으로 수정할 것
output_path = 'andmarks_output.mp4'  # 저장할 비디오 경로

#Mediapipe로 랜드마크 추출
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
                 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
                 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
                 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191,
                 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
                 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
                 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
                 415, 454, 466, 468, 473]

HAND_NUM = len(filtered_hand)
POSE_NUM = len(filtered_pose)
FACE_NUM = len(filtered_face)

hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def get_frame_landmarks(frame):
    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))
    
    def get_hands(frame):
        results_hands = hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].index == 0: 
                    all_landmarks[:HAND_NUM, :] = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]) # right
                else:
                    all_landmarks[HAND_NUM:HAND_NUM * 2, :] = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]) # left

    def get_pose(frame):
        results_pose = pose.process(frame)
        if results_pose.pose_landmarks:
            all_landmarks[HAND_NUM * 2:HAND_NUM * 2 + POSE_NUM, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark])[filtered_pose]
        
    def get_face(frame):
        results_face = face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            all_landmarks[HAND_NUM * 2 + POSE_NUM:, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark])[filtered_face]
        
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(get_hands, frame)
        executor.submit(get_pose, frame)
        executor.submit(get_face, frame)

    return all_landmarks        # 랜드마크 좌표 반환


def get_video_landmarks(video_path, start_frame=1, end_frame=-1):
    cap = cv2.VideoCapture(video_path)
    
    if start_frame <= 1:
        start_frame = 1
        
    elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        start_frame = 1
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    if end_frame < 0: 
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_landmarks = HAND_NUM * 2 + POSE_NUM + FACE_NUM
    all_frame_landmarks = np.zeros((end_frame - start_frame + 1, num_landmarks, 3))
    frame_index = 1
    
    while cap.isOpened() and frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index >= start_frame:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_landmarks = get_frame_landmarks(frame)
            all_frame_landmarks[frame_index - start_frame] = frame_landmarks

        frame_index += 1

    cap.release()
    hands.reset()
    pose.reset()
    face_mesh.reset()
    return all_frame_landmarks  # (f, 180, 3) 형태

def draw_landmarks(input_path, output_path, video_landmarks, start_frame=1, end_frame=-1):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if start_frame <= 1:
        start_frame = 1
    elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        start_frame = 1
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    frame_index = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index >= start_frame and frame_index <= end_frame:
            frame_landmarks = video_landmarks[frame_index - start_frame]
            landmarks = [(int(x * width), int(y * height)) for x, y, _ in frame_landmarks] 
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            out.write(frame)
        else:
            # out.write(frame) # Enable if you want the full video
            pass
        frame_index += 1

    cap.release()
    out.release()


labels_dict = np.load('labels.npz', allow_pickle=True)

""" 
landmarks_dict = np.load('landmarks_V3.npz', allow_pickle=True)
with open('WLASL_parsed_data.json', 'r') as json_file:
    data = json.load(json_file)

    landmarks = (
    [x for x in filtered_hand] +
    [x + HAND_NUM for x in filtered_hand] +
    [x + HAND_NUM * 2 for x in filtered_pose] +
    [x + HAND_NUM * 2 + POSE_NUM for x in filtered_face]
)

# 12-14.h5 모델에서 사용된 비디오 전처리 함수
def sequences(X, Y, length=30, step=1, pad=0):
    X_sequences = []
    Y_sequences = []

    for inputs, label in zip(X, Y):
        num = inputs.shape[0]

        if num < length:
            padding = length - num
            inputs = np.pad(
            inputs, ((0, padding), (0, 0), (0, 0)),
            mode='constant', constant_values=pad
            )
            num = length

        for start in range(0, num - length + 1, step):
            end = start + length
            sequence = inputs[start:end]
            X_sequences.append(sequence)
            Y_sequences.append(label)

    X_sequences = np.array(X_sequences)
    Y_sequences = np.array(Y_sequences)
    return X_sequences, Y_sequences

def padding(X, Y, length=None, pad=0):
    if length is None:
        length = max(len(x) for x in X)
    
    X_padded = []
    for x in X:
        if len(x) > length:
            X_padded.append(x[:length]) #truncate
        else:
            pad_length = length - len(x)
            X_padded.append(np.pad(
                x, ((0, pad_length), (0, 0), (0, 0)),
                mode='constant', constant_values=pad
            ))
            
    X_padded = np.array(X_padded)
#     Y = np.array(Y)
    return X_padded, Y

def skipping(landmarks, desired_frames,mode='floor'):
    frames_num = landmarks.shape[0]
    if mode == 'floor':
        skip_factor = math.floor(frames_num / desired_frames)
    elif mode == 'ceil':
        skip_factor = math.ceil(frames_num / desired_frames)
    skipped_landmarks = []

    for i in range(0, frames_num, skip_factor):
        skipped_landmarks.append(landmarks[i])
        if len(skipped_landmarks)==desired_frames:
            break

    return np.array(skipped_landmarks)

def cloning(landmarks, desired_frames):
    
    frames_num = landmarks.shape[0]
    repeat_factor = math.ceil(desired_frames / frames_num)
    
    cloned_list = np.repeat(landmarks, repeat_factor, axis=0)
    cloned_list = cloned_list[:desired_frames]
    return cloned_list

def clone_skip(landmarks_array,desired_frames):
    reshaped_landmarks = []
    for landmarks in landmarks_array:
        frames_number = landmarks.shape[0]
        
        if frames_number == desired_frames:
            reshaped_landmarks.append(landmarks)
        elif frames_number < desired_frames:
            reshaped_landmarks.append(cloning(landmarks,desired_frames))
        elif frames_number > desired_frames:
            reshaped_landmarks.append(skipping(landmarks,desired_frames))
    return np.array(reshaped_landmarks) """


# 200 프레임보다 작은 비디오에 대해서만 처리 중입니다
# 영상 슬라이딩 방법 필요
def predict_landmarks(model, video_landmarks):   
    num_frames = video_landmarks.shape[0]   # 비디오 랜드마크의 프레임 수 확인,  프레임 수가 200 미만일 경우 패딩 추가

    if num_frames < 200:
        padding = np.zeros((200 - num_frames, 180, 3))  # 0으로 패딩
        padded_landmarks = np.vstack((video_landmarks, padding))
    else:
        padded_landmarks = video_landmarks[:200]  

    reshaped_landmarks = padded_landmarks.reshape(1, 200, 180, 3)
    predictions = model.predict(reshaped_landmarks)

    return predictions


video_landmarks = get_video_landmarks(video_path)   # 비디오에서 랜드마크 추출 **
""" draw_landmarks(video_path, output_path, video_landmarks) # 랜드마크가 그려진 영상 저장 """

predictions = predict_landmarks(model, video_landmarks)
# print(predictions)

# 가장 높은 확률을 가진 클래스 인덱스 찾기
predicted_class_index = np.argmax(predictions, axis=1)
labels_array = list(labels_dict.keys())  # 키를 리스트로 변환

predicted_label = labels_array[predicted_class_index[0]]
print("Predicted label:", predicted_label)
