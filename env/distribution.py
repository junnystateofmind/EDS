import json
import numpy as np
from patient import Emergence_Patient, Naive_Patient, Nylon_Patient


def generate_patient_data(lam, num_patients):
    # prob을 생성하기 위한 data
    data = [475 + 75 + 80 + 109 + 45, 387]
    data = np.array(data)
    prob = data / data.sum()
    # naive, nylon 환자 구분
    naive_percent = 0.8
    prob = np.array([prob[0], prob[1] * naive_percent, prob[1] * (1 - naive_percent)])

    patient_data = []
    arrival_times = np.cumsum(np.random.poisson(lam, num_patients))
    patient_types = np.random.choice(['emergence', 'naive', 'nylon'], size=num_patients, p=prob)

    for i in range(num_patients):
        register_id = int(i + 1)  # int64를 일반 int로 변환
        arrival_time = int(arrival_times[i])  # int64를 일반 int로 변환
        patient_type = patient_types[i]

        if patient_type == 'emergence':
            patient = Emergence_Patient(register_id, arrival_time)
        elif patient_type == 'naive':
            patient = Naive_Patient(register_id, arrival_time)
        else:
            patient = Nylon_Patient(register_id, arrival_time)

        patient_data.append({
            "register_id": int(patient.get_register_id()),
            "arrival_time": int(patient.arrival_time),
            "type": patient_type,
            "deadline": int(patient.get_deadline()) if not np.isinf(patient.get_deadline()) else "inf",
            "treatment_time": float(patient.get_treatment_time()),
            "emergency_status": patient.get_emergency_status()
        })

    return patient_data


# Parameters
lam = 10  # 평균 도착 시간
num_patients = 100  # 생성할 환자 수

# Generate patient data
patient_data = generate_patient_data(lam, num_patients)

# Save to JSON file
with open('patient_data.json', 'w') as f:
    json.dump(patient_data, f, indent=4)

# 확인을 위해 JSON 파일 내용을 출력
with open('patient_data.json', 'r') as f:
    data = f.read()
    print(data)