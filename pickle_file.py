import os
import cv2
import numpy as np
import pickle



# Define directories
data1 = "NEW DATA 4 JUNE 2024/REACH"
data2 = "NEW DATA 4 JUNE 2024/GRASP"
data3 = "NEW DATA 4 JUNE 2024/PICK"
data4 = "NEW DATA 4 JUNE 2024/place"
data5 = "NEW DATA 4 JUNE 2024/RELEASE"
data6 = "NEW DATA 4 JUNE 2024/RETRACT  NEW/retract new"
data7 = "NEW DATA 4 JUNE 2024/TILT/tilt"
# data8 = "NEW DATA 4 JUNE 2024/10_mix"

data_file = 'NEW_DATA_4_JUNE(1).pkl'

# Function to extract frames from videos
def extract_frames(video_dir, label, num_frames=30):
    frame_sequences = []
    video_files = os.listdir(video_dir)
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (224, 224))  # Resize frame to match model input size
            frames.append(frame)
        if len(frames) == num_frames:
            frame_sequences.append([frames, label])
        cap.release()
    return frame_sequences

if not os.path.exists(data_file):
    # Extract frames for each subtask
    reach_sequences = extract_frames(data1, label=0)
    print(len(reach_sequences))
    retract_sequences = extract_frames(data2, label=1)
    print(len(retract_sequences))
    tilt_sequences = extract_frames(data3, label=2)
    print(len(tilt_sequences))
    release_sequences = extract_frames(data4, label=3)
    print(len(release_sequences))
    grasp_sequences = extract_frames(data5, label=4)
    print(len(grasp_sequences))
    place_sequences = extract_frames(data6, label=5)
    print(len(place_sequences))
    pick_sequences = extract_frames(data7, label=6)
    print(len(pick_sequences))
    # data_sequences = extract_frames(data8, label=7)
    # print(len(data_sequences))

    # Combine frames and labels
    sequences = reach_sequences + retract_sequences + tilt_sequences + release_sequences + grasp_sequences + place_sequences + pick_sequences

    np.random.shuffle(sequences)
    frames, labels = zip(*sequences)
    frames = np.array(frames)
    labels = np.array(labels)

    # Split data into training, validation, and test sets
    split_train = int(0.8 * len(frames))
    split_val = int(0.9 * len(frames))
    train_frames, train_labels = frames[:split_train], labels[:split_train]
    val_frames, val_labels = frames[split_train:split_val], labels[split_train:split_val]
    test_frames, test_labels = frames[split_val:], labels[split_val:]

    # Data Preprocessing
    train_frames = train_frames / 255.0  # Normalize pixel values
    val_frames = val_frames / 255.0
    test_frames = test_frames / 255.0

    # Save data to pickle file
    with open(data_file, 'wb') as f:
        pickle.dump((train_frames, train_labels, val_frames, val_labels, test_frames, test_labels), f)
else:
    # Load data from pickle file
    with open(data_file, 'rb') as f:
        train_frames, train_labels, val_frames, val_labels, test_frames, test_labels = pickle.load(f)

print(f'Training data shape: {train_frames.shape}')
print(f'Validation data shape: {val_frames.shape}')
print(f'Test data shape: {test_frames.shape}')
