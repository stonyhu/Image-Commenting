
class ExpAvgMeter:
    """Keep exponential weighted averages."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)

    def get(self):
        return self.value

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)