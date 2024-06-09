import numpy as np
import simpy
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from patient import Patient, Emergence_Patient, Naive_Patient, Nylon_Patient


class EmergencyRoom:
    def __init__(self, env, num_doctors):
        self.env = env
        self.doctor = simpy.PriorityResource(env, num_doctors)
        self.queue = []  # 치료 대기 큐
        self.treatment_processes = []  # 현재 치료 중인 프로세스 리스트
        self.logs = []  # 로그 데이터 수집

    def add_patient(self, patient):
        heapq.heappush(self.queue, (patient.get_deadline(), patient))
        self.logs.append((self.env.now, 'ADD', patient.get_register_id(), patient.get_deadline(), ''))

    def treat_patient(self, patient):
        with self.doctor.request(priority=patient.get_deadline()) as req:
            yield req

            treatment_time = patient.get_treatment_time()
            start_time = self.env.now

            while treatment_time > 0:
                if patient.get_emergency_status():
                    yield self.env.timeout(treatment_time)
                    treatment_time = 0
                else:
                    remaining_treatment = min(treatment_time, 10)
                    yield self.env.timeout(remaining_treatment)
                    treatment_time -= remaining_treatment

                    if treatment_time > 0:
                        self.logs.append((self.env.now, 'PAUSE', patient.get_register_id(), patient.get_deadline(), ''))
                        with self.doctor.request(priority=patient.get_deadline()) as new_req:
                            yield new_req
                            self.logs.append(
                                (self.env.now, 'RESUME', patient.get_register_id(), patient.get_deadline(), ''))

            if self.env.now > patient.get_deadline():
                patient.set_alive(False)
                self.logs.append((self.env.now, 'DEAD', patient.get_register_id(), patient.get_deadline(), ''))
            else:
                patient.set_alive(True)
                self.logs.append((self.env.now, 'DONE', patient.get_register_id(), patient.get_deadline(), ''))

    def check_deadlines(self):
        while True:
            now = self.env.now
            while self.queue and self.queue[0][1].get_deadline() < now:
                _, patient = heapq.heappop(self.queue)
                patient.set_alive(False)
                self.logs.append((now, 'DEAD', patient.get_register_id(), patient.get_deadline(), ''))
            yield self.env.timeout(1)  # 1분마다 확인


def patient_generator(env, er):
    register_id = 1
    while True:
        arrival_time = env.now
        patient_type = np.random.choice(['emergence', 'naive', 'nylon'], p=[0.1, 0.8, 0.1])
        if patient_type == 'emergence':
            patient = Emergence_Patient(register_id, arrival_time)
        elif patient_type == 'naive':
            patient = Naive_Patient(register_id, arrival_time)
        else:
            patient = Nylon_Patient(register_id, arrival_time)

        er.add_patient(patient)
        env.process(er.treat_patient(patient))
        register_id += 1
        yield env.timeout(np.random.exponential(10))


def run_simulation():
    env = simpy.Environment()
    er = EmergencyRoom(env, num_doctors=8)
    env.process(patient_generator(env, er))
    env.process(er.check_deadlines())
    env.run(until=1000)
    return er.logs


def visualize_logs(logs):
    df = pd.DataFrame(logs, columns=['Time', 'Event', 'PatientID', 'Deadline', 'Note'])
    fig, ax = plt.subplots(figsize=(12, 6))

    for event_type, group in df.groupby('Event'):
        ax.scatter(group['Time'], group['PatientID'], label=event_type, alpha=0.6)

    ax.set_xlabel('Time')
    ax.set_ylabel('Patient ID')
    ax.set_title('Emergency Room Events Over Time')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    logs = run_simulation()
    visualize_logs(logs)