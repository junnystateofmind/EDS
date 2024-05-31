import numpy as np
import simpy
import heapq
from patient import Patient, Emergence_Patient, Naive_Patient, Nylon_Patient

class EmergencyRoom:
    def __init__(self, env, num_doctors):
        self.env = env
        self.doctor = simpy.PriorityResource(env, num_doctors)
        self.queue = []

    def add_patient(self, patient):
        heapq.heappush(self.queue, (patient.get_deadline(), patient))

    def treat_patient(self, patient):
        with self.doctor.request(priority=patient.get_deadline()) as req:
            yield req
            yield self.env.timeout(patient.get_treatment_time())
            if self.env.now > patient.get_deadline():
                patient.set_alive(False)
                print(f"환자 {patient.get_register_id()}가 {self.env.now}에 사망했습니다. 데드라인은 {patient.get_deadline()}입니다.")
            else:
                patient.set_alive(True)
                print(f"환자 {patient.get_register_id()}가 {self.env.now}에 치료를 마쳤습니다. 데드라인은 {patient.get_deadline()}이었습니다.")

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
        # 10분마다 환자가 도착
        yield env.timeout(np.random.exponential(10))

def run_simulation():
    env = simpy.Environment()
    er = EmergencyRoom(env, num_doctors=4)
    env.process(patient_generator(env, er))
    env.run(until=1000)

if __name__ == "__main__":
    run_simulation()