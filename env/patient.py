import numpy as np
from functools import total_ordering

@total_ordering
class Patient:
    def __init__(self, register_id, arrival_time):
        self.register_id = register_id
        self.emergency_status = False
        self.deadline = 180
        self.alive = True
        self.arrival_time = arrival_time
        self.treatment_time = 0

    def set_emergency(self):
        self.emergency_status = True

    def set_deadline(self, deadline):
        self.deadline = deadline

    def set_alive(self, alive):
        self.alive = alive

    def set_treatment_time(self, treatment_time):
        self.treatment_time = treatment_time

    def get_register_id(self):
        return self.register_id

    def get_emergency_status(self):
        return self.emergency_status

    def get_deadline(self):
        return self.deadline

    def get_alive(self):
        return self.alive

    def get_treatment_time(self):
        return self.treatment_time

    def __lt__(self, other):
        return self.deadline < other.deadline

    def __eq__(self, other):
        return self.deadline == other.deadline

    def __str__(self):
        return "Register ID: {}, Emergency: {}, Deadline: {}, Alive: {}, Treatment Time: {}".format(
            self.register_id, self.emergency_status, self.deadline, self.alive, self.treatment_time)

class Emergence_Patient(Patient):
    def __init__(self, register_id, arrival_time):
        super().__init__(register_id, arrival_time)
        self.set_emergency()
        self.set_deadline(arrival_time + int(np.random.exponential(scale=60)))  # 지수 분포로 데드라인 설정
        self.set_treatment_time(int(np.random.exponential(scale=30)))  # 지수 분포로 치료 시간 생성

    def __str__(self):
        return "Register ID: {}, Emergency: {}, Deadline: {}, Alive: {}, Treatment Time: {}".format(
            self.register_id, self.emergency_status, self.deadline, self.alive, self.treatment_time)

class Naive_Patient(Patient):
    def __init__(self, register_id, arrival_time):
        super().__init__(register_id, arrival_time)
        self.set_deadline(np.inf)  # 무한대 데드라인
        self.set_treatment_time(int(np.random.normal(loc=30, scale=10)))  # 정규 분포로 치료 시간 생성

    def __str__(self):
        return "Register ID: {}, Emergency: {}, Deadline: {}, Alive: {}, Treatment Time: {}".format(
            self.register_id, self.emergency_status, self.deadline, self.alive, self.treatment_time)

class Nylon_Patient(Patient):
    def __init__(self, register_id, arrival_time):
        super().__init__(register_id, arrival_time)
        self.set_emergency()
        self.set_deadline(np.inf)  # 무한대 데드라인
        self.set_treatment_time(int(np.random.exponential(scale=10)))  # 지수 분포로 치료 시간 생성

    def __str__(self):
        return "Register ID: {}, Emergency: {}, Deadline: {}, Alive: {}, Treatment Time: {}".format(
            self.register_id, self.emergency_status, self.deadline, self.alive, self.treatment_time)