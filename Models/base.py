

class BaseModel:
    def __init__(self, training_data, full_data):
        self.training_data = training_data
        self.full_data = full_data
        self.model = None

    def train(self):
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self):
        raise NotImplementedError("Subclasses must implement predict()")

    def evaluate(self):
        raise NotImplementedError("Subclasses must implement evaluate()")

