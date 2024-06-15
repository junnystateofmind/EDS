import simpy
import json
import pandas as pd
import matplotlib.pyplot as plt
from patient import Emergence_Patient, Naive_Patient, Nylon_Patient
import heapq
import numpy as np

class EmergencyRoom:
    def __init__(self, env, num_doctors):
        self.env = env
        self.num_doctors = num_doctors
        self.doctor = simpy.PriorityResource(env, num_doctors)
        self.queue = []
        self.treatment_processes = []
        self.logs = []
        self.bed_logs = {i: [] for i in range(num_doctors)}

    def add_patient(self, patient):
        heapq.heappush(self.queue, (patient.get_deadline(), patient))
        self.logs.append((self.env.now, 'ARRIVE', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))

    def treat_patient(self, patient):
        with self.doctor.request(priority=patient.get_deadline()) as req:
            yield req

            treatment_time = patient.get_treatment_time()
            start_time = self.env.now
            self.logs.append((start_time, 'START', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
            self._log_bed_status(start_time, 'OCCUPIED', patient.get_register_id(), patient.__class__.__name__)

            while treatment_time > 0:
                if patient.get_emergency_status():
                    yield self.env.timeout(treatment_time)
                    treatment_time = 0
                else:
                    remaining_treatment = min(treatment_time, 10)
                    yield self.env.timeout(remaining_treatment)
                    treatment_time -= remaining_treatment

                    if treatment_time > 0:
                        self.logs.append((self.env.now, 'PAUSE', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                        self._log_bed_status(self.env.now, 'PAUSE', patient.get_register_id(), patient.__class__.__name__)
                        with self.doctor.request(priority=patient.get_deadline()) as new_req:
                            yield new_req
                            self.logs.append((self.env.now, 'RESUME', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                            self._log_bed_status(self.env.now, 'RESUME', patient.get_register_id(), patient.__class__.__name__)

            if self.env.now > patient.get_deadline() and patient.get_deadline() != np.inf:
                patient.set_alive(False)
                self.logs.append((self.env.now, 'DEAD', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(self.env.now, 'DEAD', patient.get_register_id(), patient.__class__.__name__)
            else:
                patient.set_alive(True)
                self.logs.append((self.env.now, 'DONE', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(self.env.now, 'DONE', patient.get_register_id(), patient.__class__.__name__)

    def _log_bed_status(self, time, status, patient_id, patient_type):
        for bed_id in range(self.num_doctors):
            self.bed_logs[bed_id].append((time, status, patient_id, patient_type))

    def check_deadlines(self):
        while True:
            now = self.env.now
            while self.queue and self.queue[0][1].get_deadline() < now and self.queue[0][1].get_deadline() != np.inf:
                _, patient = heapq.heappop(self.queue)
                patient.set_alive(False)
                self.logs.append((now, 'DEAD', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(now, 'DEAD', patient.get_register_id(), patient.__class__.__name__)
            yield self.env.timeout(1)  # 1분마다 확인

    def finalize_logs(self):
        now = self.env.now
        while self.queue:
            _, patient = heapq.heappop(self.queue)
            if patient.get_alive():
                self.logs.append((now, 'UNFINISHED', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(now, 'UNFINISHED', patient.get_register_id(), patient.__class__.__name__)

def load_patient_data(file_path):
    with open(file_path, 'r') as f:
        patient_data = json.load(f)
    return patient_data

def patient_generator(env, er, patient_data):
    for patient_info in patient_data:
        register_id = patient_info['register_id']
        arrival_time = patient_info['arrival_time']
        patient_type = patient_info['type']

        if patient_type == 'emergence':
            patient = Emergence_Patient(register_id, arrival_time)
        elif patient_type == 'naive':
            patient = Naive_Patient(register_id, arrival_time)
        else:
            patient = Nylon_Patient(register_id, arrival_time)

        yield env.timeout(arrival_time - env.now)  # 다음 환자가 도착할 때까지 대기
        er.add_patient(patient)
        env.process(er.treat_patient(patient))

def run_simulation(patient_data):
    env = simpy.Environment()
    er = EmergencyRoom(env, num_doctors=4)
    env.process(patient_generator(env, er, patient_data))
    env.process(er.check_deadlines())
    env.run(until=1000)
    er.finalize_logs()  # 시뮬레이션 종료 시 미처리 환자 기록
    return er.logs, er.bed_logs

def visualize_logs(logs, bed_logs):
    df = pd.DataFrame(logs, columns=['Time', 'Event', 'PatientID', 'Deadline', 'EmergencyStatus', 'PatientType'])
    bed_df = pd.DataFrame([(bed_id, time, status, patient_id, patient_type) for bed_id, logs in bed_logs.items() for time, status, patient_id, patient_type in logs], columns=['BedID', 'Time', 'Status', 'PatientID', 'PatientType'])

    fig, ax = plt.subplots(figsize=(12, 6))

    event_colors = {
        'ARRIVE': 'blue',
        'START': 'green',
        'PAUSE': 'orange',
        'RESUME': 'purple',
        'DONE': 'black',
        'DEAD': 'red',
        'UNFINISHED': 'grey'
    }

    for patient_id, group in df.groupby('PatientID'):
        times = group['Time']
        events = group['Event']
        start_time = group[group['Event'] == 'ARRIVE']['Time'].values[0]

        if 'DONE' in events.values:
            end_time = group[group['Event'] == 'DONE']['Time'].values[0]
        elif 'DEAD' in events.values:
            end_time = group[group['Event'] == 'DEAD']['Time'].values[0]
        elif 'UNFINISHED' in events.values:
            end_time = group[group['Event'] == 'UNFINISHED']['Time'].values[0]
        else:
            continue  # 'DONE', 'DEAD', 'UNFINISHED' 이벤트가 없으면 건너뜀

        # 선 그리기
        ax.plot([start_time, end_time], [patient_id, patient_id], 'k-', alpha=0.5)

        # 마커 그리기
        for event, color in event_colors.items():
            if event in events.values:
                event_times = group[group['Event'] == event]['Time']
                ax.scatter(event_times, [patient_id] * len(event_times), color=color, label=event if patient_id == 1 else "")

    ax.set_xlabel('Time')
    ax.set_ylabel('Patient ID')
    ax.set_title('Emergency Room Events Over Time')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()

    # 로그를 CSV 파일로 저장
    df.to_csv('simulation_logs.csv', index=False)
    bed_df.to_csv('bed_logs.csv', index=False)
    print("Logs saved to simulation_logs.csv")
    print("Bed logs saved to bed_logs.csv")

    # 사망한 환자 로그 출력
    dead_patients = df[df['Event'] == 'DEAD']
    print("\nDead Patients Logs:")
    print(dead_patients[['Time', 'PatientID', 'PatientType', 'Deadline']].to_string(index=False))

if __name__ == "__main__":
    patient_data = load_patient_data('patient_data.json')
    logs, bed_logs = run_simulation(patient_data)
    visualize_logs(logs, bed_logs)