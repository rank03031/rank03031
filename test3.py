import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor


# 모델 로드
model = tf.keras.models.load_model('12-14.h5')

video_path = '' # 입력할 비디오
output_path = 'translated_video.mp4'  # 저장할 비디오 경로
landmark_output_path = 'landmarks_output.mp4'

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

# 랜드마크 추출
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

    return all_landmarks        # 랜드마크 좌표 반환(2차원)


def get_video_landmarks(video_path, start_frame=1, end_frame=-1):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame <= 1:
        start_frame = 1
        
    elif start_frame > frame_count:
        start_frame = 1
        end_frame = frame_count
        
    if end_frame < 0: 
        end_frame = frame_count

    frame_index = 1
    
    sign_sequences = np.zeros((frame_count, frame_count, 180, 3))  # 수어 동작을 저장할 리스트
    current_sign = np.zeros((frame_count, 180, 3)) 
    in_sign = False  # 현재 수어 동작 중인지 여부
    cnt = 1
    
    while cap.isOpened() and frame_index <= end_frame:
        
        ret, frame = cap.read()
        h, w, _ = frame.shape
        if not ret:
            break
        if frame_index >= start_frame:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_landmarks = get_frame_landmarks(frame)
            START_THRESHOLD = h * 0.95
 
            if frame_landmarks[0][1]:
                right_wrist_y = frame_landmarks[0][1] * h 
            else:
                right_wrist_y = h  
            if frame_landmarks[21][1]:
                left_wrist_y = frame_landmarks[21][1] * h
            else:
                left_wrist_y = h
      
            wrist_y = min(right_wrist_y, left_wrist_y)
            #print(wrist_y)
            if wrist_y < START_THRESHOLD:
                if not in_sign:
                    in_sign = True
                current_sign[frame_index - start_frame, :, :] = frame_landmarks  # 현재 수어 동작의 프레임, 랜드마크
                
            else:
                if in_sign:
                    sign_sequences[cnt, :, :, :] = current_sign # (cnt, frame, 180, 3)
                    current_sign = np.zeros((frame_count, 180, 3))
                    cnt += 1
                    in_sign = False

        frame_index += 1
   
    #print(sign_sequences)
    cap.release()
    hands.reset()
    pose.reset()
    face_mesh.reset()
    return sign_sequences, cnt



""" def draw_landmarks(input_path, output_path, video_landmarks, start_frame=1, end_frame=-1):
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
    out.release() """



""" # 팔 각도 계산 : 정확하지 않다.
def angle_product(frame, shoulder, elbow, wrist) :
    if shoulder is None or elbow is None or wrist is None:
        return False
    
    h, w, _ = frame.shape
    
    shoulder_coords = (int(shoulder[0] * w), int(shoulder[1] * h))
    elbow_coords = (int(elbow[0] * w), int(elbow[1] * h))
    wrist_coords = (int(wrist[0] * w), int(wrist[1] * h)) 
    
    AB = np.array([elbow_coords[0] - shoulder_coords[0], elbow_coords[1] - shoulder_coords[1]])
    BC = np.array([wrist_coords[0] - elbow_coords[0], wrist_coords[1] - elbow_coords[1]])
    
    # 벡터의 길이를 계산
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC) 
    
    # 벡터의 길이가 0인 경우 체크
    if norm_AB == 0 or norm_BC == 0:
        return False 
    
    dot_product = np.dot(AB, BC)
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC)
    cos_theta = dot_product / (norm_AB * norm_BC)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle)  
    
    print(angle_degrees)
    
    return angle_degrees """


#각 시퀀스별로 예측 수행
def predict_landmarks(model, sign_sequences, cnt):  
    predictions_list = []
    length = 200  
    frame_count = len(sign_sequences[0])
    for i in range(cnt-1):            #시퀀스 개수만큼 수행
        fcount = -1
        num = 0
        while (np.any(sign_sequences[i][num]) != 0):
            fcount += 1        # 각 동작의 프레임 수 확인
            num += 1

        if fcount < length and frame_count < length:
            reshaped_sequence = np.zeros((1, length, 180, 3))
            padded_seq = padding(sign_sequences[i], frame_count, length)  # 패딩 추가
            reshaped_sequence[0 , :, :, : ] = padded_seq
            
            predict = model.predict(reshaped_sequence)  
            predicted_indices = np.argmax(predict, axis=1)
            predicted_label = [labels_array[i] for i in predicted_indices]  # 레이블 변환
            print("Predicted labels:", predicted_label)
            predictions_list.append(predicted_label)  #예측 리스트 저장
            
        elif fcount < length and frame_count > length:
            reshaped_sequence = np.zeros((1, length, 180, 3))
            reshaped_sequence[0 , :, :, : ] = sign_sequences[i][:length]  
            
            predict = model.predict(reshaped_sequence)  
            predicted_indices = np.argmax(predict, axis=1)
            predicted_label = [labels_array[i] for i in predicted_indices]
            print("Predicted labels:", predicted_label)
            predictions_list.append(predicted_label)  
            
        else :
            best_prediction = double_sequence(sign_sequences[i], fcount)
            predicted_label = labels_array[best_prediction]  
            predictions_list.append(predicted_label)


    predictions = np.array(predictions_list)  
    #print("Raw predictions shape:", predictions.shape)  
    return predictions



def double_sequence(sequence, fcount, length=200):
    sequences = np.zeros((fcount-length+1, length, 180, 3))
        # fcount가 length보다 클 경우에만 실행
        # sequence의 길이    
    for start in range(fcount-length+1):
            # start 인덱스에서 length 길이만큼 자른 부분을 추가
        end = start + length
        sequences[start, :, :, :] = sequence[start:end]
    
    predictions = model.predict(sequences)

    best_index = np.argmax(np.max(predictions, axis=1))
    best_prediction = np.argmax(predictions[best_index])  # 해당 시퀀스에서 가장 높은 확률을 가진 클래스의 인덱스
    
    return best_prediction


def padding(sequence, frame_count, length):  
    pad_length = length - frame_count
    return np.pad(sequence, ((0, pad_length), (0, 0), (0, 0)), mode='constant', constant_values=0)


def overlay_subtitle(input_path, output_path, predictions, start_frame=1, end_frame=-1, font_scale=1, color=(255, 255, 255), thickness=2):
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
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    in_sign = False  # 현재 수어 동작 중인지 여부
    d = 0
    subtitle_text = ""   
            
    frame_index = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_landmarks = get_frame_landmarks(frame)
        START_THRESHOLD = height * 0.95
        
        # 오른쪽 손목 y 좌표 계산
        if frame_landmarks[0][1]:
            right_wrist_y = frame_landmarks[0][1] * height 
        else:
            right_wrist_y = height  
        
        # 왼쪽 손목 y 좌표 계산
        if frame_landmarks[21][1]:
            left_wrist_y = frame_landmarks[21][1] * height
        else:
            left_wrist_y = height

        # 두 손목 중 더 낮은 y 좌표 선택
        wrist_y = min(right_wrist_y, left_wrist_y)
        
        # 수어동작중이면 텍스트 표시
        if wrist_y < START_THRESHOLD:
            if not in_sign: 
                in_sign = True
                subtitle_text = predictions[d][0] if d < len(predictions) and len(predictions[d]) > 0 else ""
                text_size = cv2.getTextSize(subtitle_text, font, font_scale, thickness)[0]  # 폰트 설정
                text_x = (width - text_size[0]) // 2 
                text_y = height - 10
                d += 1
            cv2.putText(frame, subtitle_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            out.write(frame)
                
        else:
            in_sign = False
            subtitle_text = ""
            out.write(frame)


        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    


labels_dict = np.load('labels.npz', allow_pickle=True)
labels_array = list(labels_dict.keys())  # 키를 리스트로 변환



def video_translate(video_path):
    #4차원 배열과 시퀀스 개수
    sign_sequences, cnt = get_video_landmarks(video_path)   # 비디오에서 랜드마크 추출 ** 시작점
    """ draw_landmarks(video_path, landmark_output_path, video_landmarks) # 랜드마크가 그려진 영상 저장 """

    predictions = predict_landmarks(model, sign_sequences, cnt)
    overlay_subtitle(video_path, output_path, predictions)
    # print(predictions)
