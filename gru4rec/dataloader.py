import numpy as np

class SessionParallelLoader:
    def __init__(self, sequences, batch_size):
        self.batch_size = batch_size

        # 길이 2 이상 세션만 사용
        self.sequences = [s for s in sequences if len(s) >= 2]

        # 긴 세션부터 정렬 (논문 방식)
        order = np.argsort([len(s) for s in self.sequences])[::-1]
        self.sequences = [self.sequences[i] for i in order]

        self.n_sessions = len(self.sequences)

    def __iter__(self):
        self.session_idx = self.batch_size
        self.active_sessions = self.sequences[:self.batch_size]
        self.positions = np.zeros(len(self.active_sessions), dtype=np.int64)
        return self

    def __next__(self):
        if len(self.active_sessions) == 0:
            raise StopIteration

        B = len(self.active_sessions)

        x = np.zeros(B, dtype=np.int64)
        y = np.zeros(B, dtype=np.int64)
        reset = np.zeros(B, dtype=bool)

        for i in range(B):
            seq = self.active_sessions[i]
            pos = self.positions[i]

            # 정상 step: 항상 x, y 존재
            x[i] = seq[pos]
            y[i] = seq[pos + 1]

            self.positions[i] += 1

            # session 종료 직전이면 다음 step에서 교체
            if self.positions[i] + 1 >= len(seq):
                if self.session_idx < self.n_sessions:
                    self.active_sessions[i] = self.sequences[self.session_idx]
                    self.positions[i] = 0
                    self.session_idx += 1
                    reset[i] = True   # ⭐ 새 session 시작
                else:
                    # 더 넣을 session 없으면 batch slot 제거
                    self.active_sessions[i] = None

        # batch slot 유지 (None 제거)
        keep = [i for i, s in enumerate(self.active_sessions) if s is not None]
        self.active_sessions = [self.active_sessions[i] for i in keep]
        self.positions = self.positions[keep]

        return x[keep], y[keep], reset[keep]

    def __len__(self):
        # tqdm용 근사치
        return sum(len(s) - 1 for s in self.sequences) // self.batch_size
