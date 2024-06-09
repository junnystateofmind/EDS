import numpy as np

class first_diagnosis: # 초진하면서, 환자의 상태를 초기화하는 클래스
    def __init__(self):
        # 환자의 확률 분포 설정
        self.p_emergency = 0.2
        self.emergency_distribution = np.random.poisson(self.p_emergency, 1) # 응급 확률 분포, 포아송 분포로 설정, 포아송 분포는 이산 확률 분포로, 단위 시간 안에 특정 사건이 발생하는 횟수를 표현