import simpy
import json
import pandas as pd
import matplotlib.pyplot as plt
from patient import Emergence_Patient, Naive_Patient, Nylon_Patient
import heapq
import numpy as np

class EmergencyRoom:
    def __init__(self, env, num_beds, doctor_efficiency):
        self.env = env
        self.beds = [simpy.PriorityResource(env, capacity=1) for _ in range(num_beds)]
        self.bed_patient_map = {}  # 병상과 환자를 매핑
        self.queue = []
        self.logs = []
        self.bed_logs = {i: [] for i in range(num_beds)}
        self.queue_logs = []
        self.dead_patients = set()
        self.doctor_efficiency = doctor_efficiency

    def add_patient(self, patient):
        if patient in (p[1] for p in self.queue):
            return

        if patient.get_emergency_status():
            for bed_id, bed in enumerate(self.beds):
                if bed.count > 0 and bed_id in self.bed_patient_map:
                    current_patient = self.bed_patient_map[bed_id]
                    if not current_patient.get_emergency_status():
                        self.logs.append((self.env.now, 'INTERRUPT', current_patient.get_register_id(),
                                          current_patient.get_deadline(),
                                          current_patient.get_emergency_status(), current_patient.__class__.__name__))
                        print(f"Interrupting patient {current_patient.get_register_id()} in bed {bed_id} for emergency patient {patient.get_register_id()}")
                        heapq.heappush(self.queue, (current_patient.get_deadline(), current_patient))
                        bed.users.clear()
                        self.bed_patient_map.pop(bed_id, None)  # 매핑 제거
                        break

        heapq.heappush(self.queue, (patient.get_deadline(), patient))
        self.logs.append((self.env.now, 'ARRIVE', patient.get_register_id(), patient.get_deadline(),
                          patient.get_emergency_status(), patient.__class__.__name__))
        print(f"Patient {patient.get_register_id()} arrived at {self.env.now}")
        self.log_queue_status()

        if patient.get_emergency_status():
            for i in range(len(self.beds)):
                if self.beds[i].count < self.beds[i].capacity:
                    self.env.process(self.treat_patient(patient, i))
                    break

    def treat_patient(self, patient, bed_id):
        if patient.get_register_id() in self.dead_patients:
            return

        with self.beds[bed_id].request(priority=patient.get_deadline()) as req:
            yield req

            self.bed_patient_map[bed_id] = patient

            if self.env.now > patient.get_deadline() and patient.get_deadline() != np.inf:
                patient.set_alive(False)
                self.logs.append((self.env.now, 'DEAD', patient.get_register_id(), patient.get_deadline(),
                                  patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(self.env.now, 'DEAD', bed_id, patient.get_register_id(),
                                     patient.__class__.__name__)
                self.dead_patients.add(patient.get_register_id())
                print(f"Patient {patient.get_register_id()} died at {self.env.now} in bed {bed_id} (start check)")
                self.bed_patient_map.pop(bed_id, None)  # 매핑 제거
                return

            treatment_time = patient.get_treatment_time()
            start_time = self.env.now
            patient.set_start_time(start_time)
            self.logs.append((start_time, 'START', patient.get_register_id(), patient.get_deadline(),
                              patient.get_emergency_status(), patient.__class__.__name__))
            self._log_bed_status(start_time, 'OCCUPIED', bed_id, patient.get_register_id(), patient.__class__.__name__)
            print(f"Starting treatment for patient {patient.get_register_id()} in bed {bed_id} at {self.env.now}")

            while treatment_time > 0:
                remaining_treatment = min(treatment_time, 10 * self.doctor_efficiency)
                yield self.env.timeout(remaining_treatment / self.doctor_efficiency)
                treatment_time -= remaining_treatment

                if treatment_time > 0:
                    self.logs.append((self.env.now, 'PAUSE', patient.get_register_id(), patient.get_deadline(),
                                      patient.get_emergency_status(), patient.__class__.__name__))
                    self._log_bed_status(self.env.now, 'PAUSE', bed_id, patient.get_register_id(),
                                         patient.__class__.__name__)
                    print(f"Pausing treatment for patient {patient.get_register_id()} in bed {bed_id} at {self.env.now}")
                    with self.beds[bed_id].request(priority=patient.get_deadline()) as new_req:
                        yield new_req
                        self.logs.append((self.env.now, 'RESUME', patient.get_register_id(), patient.get_deadline(),
                                          patient.get_emergency_status(), patient.__class__.__name__))
                        self._log_bed_status(self.env.now, 'RESUME', bed_id, patient.get_register_id(),
                                             patient.__class__.__name__)
                        print(f"Resuming treatment for patient {patient.get_register_id()} in bed {bed_id} at {self.env.now}")

            if self.env.now <= patient.get_deadline() or patient.get_deadline() == np.inf:
                patient.set_alive(True)
                self.logs.append((self.env.now, 'DONE', patient.get_register_id(), patient.get_deadline(),
                                  patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(self.env.now, 'DONE', bed_id, patient.get_register_id(),
                                     patient.__class__.__name__)
                self.queue = [(d, p) for d, p in self.queue if p.get_register_id() != patient.get_register_id()]
                heapq.heapify(self.queue)
                self.bed_patient_map.pop(bed_id, None)  # 매핑 제거
                print(f"Finished treatment for patient {patient.get_register_id()} in bed {bed_id} at {self.env.now}")
            else:
                if patient.get_register_id() not in self.dead_patients:
                    patient.set_alive(False)
                    self.logs.append((self.env.now, 'DEAD', patient.get_register_id(), patient.get_deadline(),
                                      patient.get_emergency_status(), patient.__class__.__name__))
                    self._log_bed_status(self.env.now, 'DEAD', bed_id, patient.get_register_id(),
                                         patient.__class__.__name__)
                    self.dead_patients.add(patient.get_register_id())
                    self.bed_patient_map.pop(bed_id, None)  # 매핑 제거
                    print(f"Patient {patient.get_register_id()} died at {self.env.now} in bed {bed_id}")

            self.log_queue_status()

    def _log_bed_status(self, time, status, bed_id, patient_id, patient_type):
        if bed_id != -1:
            self.bed_logs[bed_id].append((time, status, patient_id, patient_type))

    def log_queue_status(self):
        now = self.env.now
        queue_status = [(patient.get_register_id(), patient.__class__.__name__, patient.get_deadline()) for _, patient in self.queue]
        self.queue_logs.append((now, len(self.queue), queue_status))

    def check_deadlines(self):
        while True:
            now = self.env.now
            while self.queue and self.queue[0][1].get_deadline() < now and self.queue[0][1].get_deadline() != np.inf:
                _, patient = heapq.heappop(self.queue)
                if patient.get_register_id() not in self.dead_patients:
                    patient.set_alive(False)
                    self.logs.append((now, 'DEAD', patient.get_register_id(), patient.get_deadline(),
                                      patient.get_emergency_status(), patient.__class__.__name__))
                    self._log_bed_status(now, 'DEAD', -1, patient.get_register_id(), patient.__class__.__name__)
                    self.dead_patients.add(patient.get_register_id())
                    print(f"Patient {patient.get_register_id()} died at {now} in queue")
            self.log_queue_status()
            yield self.env.timeout(1)

    def finalize_logs(self):
        now = self.env.now
        while self.queue:
            _, patient = heapq.heappop(self.queue)
            if patient.get_alive():
                self.logs.append((now, 'UNFINISHED', patient.get_register_id(), patient.get_deadline(), patient.get_emergency_status(), patient.__class__.__name__))
                self._log_bed_status(now, 'UNFINISHED', -1, patient.get_register_id(), patient.__class__.__name__)
        self.log_queue_status()

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

        yield env.timeout(arrival_time - env.now)
        er.add_patient(patient)

        for i in range(len(er.beds)):
            if er.beds[i].count < er.beds[i].capacity:
                env.process(er.treat_patient(patient, i))
                break

def run_simulation(patient_data, doctor_efficiency):
    env = simpy.Environment()
    er = EmergencyRoom(env, num_beds=32, doctor_efficiency=doctor_efficiency)
    env.process(patient_generator(env, er, patient_data))
    env.process(er.check_deadlines())
    env.run(until=1000)
    er.finalize_logs()
    return er.logs, er.bed_logs, er.queue_logs

def visualize_logs(all_logs, all_bed_logs, all_queue_logs):
    logs_df = pd.concat([pd.DataFrame(logs, columns=['Sequence', 'Time', 'Event', 'PatientID', 'Deadline', 'EmergencyStatus', 'PatientType']) for logs in all_logs])
    bed_logs_df = pd.concat([pd.DataFrame([(seq, bed_id, time, status, patient_id, patient_type) for (seq, bed_logs) in seq_bed_logs.items() for bed_id, logs in bed_logs.items() for time, status, patient_id, patient_type in logs], columns=['Sequence', 'BedID', 'Time', 'Status', 'PatientID', 'PatientType']) for seq_bed_logs in all_bed_logs])
    queue_logs_df = pd.concat([pd.DataFrame([(seq, time, length, status) for (seq, logs) in seq_queue_logs.items() for time, length, status in logs], columns=['Sequence', 'Time', 'QueueLength', 'QueueStatus']) for seq_queue_logs in all_queue_logs])

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

    for patient_id, group in logs_df.groupby('PatientID'):
        times = group['Time']
        events = group['Event']
        arrive_event = group[group['Event'] == 'ARRIVE']

        if arrive_event.empty:
            continue

        start_time = arrive_event['Time'].values[0]

        if 'DONE' in events.values:
            end_time = group[group['Event'] == 'DONE']['Time'].values[0]
        elif 'DEAD' in events.values:
            end_time = group[group['Event'] == 'DEAD']['Time'].values[0]
        elif 'UNFINISHED' in events.values:
            end_time = group[group['Event'] == 'UNFINISHED']['Time'].values[0]
        else:
            continue

        ax.plot([start_time, end_time], [patient_id, patient_id], 'k-', alpha=0.5)

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

    logs_df.to_csv('new_simulation_logs.csv', index=False)
    bed_logs_df.to_csv('new_bed_logs.csv', index=False)
    queue_logs_df.to_csv('new_queue_logs.csv', index=False)
    print("Logs saved to new_simulation_logs.csv")
    print("Bed logs saved to new_bed_logs.csv")
    print("Queue logs saved to new_queue_logs.csv")

    dead_patients = logs_df[logs_df['Event'] == 'DEAD']
    print("\nDead Patients Logs:")
    print(dead_patients[['Sequence', 'Time', 'PatientID', 'PatientType', 'Deadline']].to_string(index=False))

    final_status_df = logs_df[logs_df['Event'].isin(['DONE', 'DEAD', 'UNFINISHED'])].copy()
    final_status_df.sort_values(by=['PatientID', 'Time'], inplace=True)
    final_status_df.drop_duplicates(subset=['PatientID'], keep='last', inplace=True)
    final_status_df.to_csv('new_final_patient_status.csv', index=False)
    print("Final patient status saved to new_final_patient_status.csv")

if __name__ == "__main__":
    all_patient_data = load_patient_data('patient_data_sequences.json')
    all_logs = []
    all_bed_logs = []
    all_queue_logs = []
    for i, patient_data in enumerate(all_patient_data):
        logs, bed_logs, queue_logs = run_simulation(patient_data, doctor_efficiency=5.0)
        for log in logs:
            log.insert(0, i)
        for bed_id in bed_logs:
            for bed_log in bed_logs[bed_id]:
                bed_log.insert(0, i)
        for queue_log in queue_logs:
            queue_log.insert(0, i)
        all_logs.append(logs)
        all_bed_logs.append(bed_logs)
        all_queue_logs.append(queue_logs)
    visualize_logs(all_logs, all_bed_logs, all_queue_logs)
if __name__ == "__main__":
    all_patient_data = load_patient_data('patient_data_sequences.json')
    all_logs = []
    for patient_data in all_patient_data:
        logs, bed_logs, queue_logs = run_simulation(patient_data, doctor_efficiency=5.0)
        all_logs.append(logs)
    # 전체 로그를 사용하여 시각화 또는 분석
    # visualize_logs를 호출할 때 all_logs를 사용하여 적절히 수정