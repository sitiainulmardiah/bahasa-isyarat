import numpy as np

def normalize_landmarks(landmarks):
    """
    Menormalkan koordinat landmark relatif terhadap pergelangan tangan (landmark 0)
    dari tangan itu sendiri.
    Input: list of 21 landmark points [[x, y, z], [x, y, z], ...]
    Output: numpy array shape (21, 3)
    """
    if len(landmarks) == 0:
        return np.zeros((21, 3))
        
    # Ambil koordinat pergelangan tangan sebagai titik acuan (0,0,0)
    base_x, base_y, base_z = landmarks[0]
    
    normalized_landmarks = []
    for point in landmarks:
        # Kurangi semua titik dengan titik acuan pergelangan tangan ini
        norm_x = point[0] - base_x
        norm_y = point[1] - base_y
        norm_z = point[2] - base_z
        normalized_landmarks.append([norm_x, norm_y, norm_z])
        
    return np.array(normalized_landmarks)


def extract_two_hands(multi_hand_landmarks, multi_handedness):
    """
    Mengambil hingga 2 tangan dan menyusunnya secara KONSISTEN berdasarkan orientasi biologis:
    - Index 0-20  : SELALU Tangan Kiri (Left)
    - Index 21-41 : SELALU Tangan Kanan (Right)
    Jika salah satu tangan tidak terdeteksi, bagian indeksnya akan tetap bernilai 0.
    """
    # Siapkan array kosong (42 titik x 3 koordinat)
    combined_landmarks = np.zeros((42, 3))
    
    if not multi_hand_landmarks or not multi_handedness:
        return combined_landmarks

    for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
        if hand_idx >= 2: 
            break # Batasi maksimal 2 tangan
            
        # Ambil label tangan ('Left' atau 'Right') dari MediaPipe
        hand_label = multi_handedness[hand_idx].classification[0].label
        
        # Ekstrak koordinat mentah [x, y, z]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        
        # Normalisasi secara independen berbasis pergelangan tangan ini sendiri
        norm_landmarks = normalize_landmarks(landmarks)
        
        # Kunci posisi array secara absolut berdasarkan label biologis tangan
        if hand_label == 'Left':
            # Tangan kiri menempati index 0 sampai 20
            combined_landmarks[0:21] = norm_landmarks
        elif hand_label == 'Right':
            # Tangan kanan menempati index 21 sampai 41
            combined_landmarks[21:42] = norm_landmarks
            
    return combined_landmarks