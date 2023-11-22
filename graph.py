import matplotlib.pyplot as plot
from model import Model

class Graph:
    def __init__(self, model: Model):
        self.model = model

    def plot_normalized(self):
        data = range(len(self.model.y))

        fig, plots = plot.subplots(1, self.model.ny, squeeze=False, figsize=(15, 4))
        for i in range(self.model.ny):
            plots[0][i].plot(data, self.model.yn[:, i], label=f'Y{i+1}')
            plots[0][i].plot(data, self.model.predict_normalized[:, i], label=f'Ф{i+1}')
            plots[0][i].set_title(f"Похибка: {self.model.error_normalized[i]:.4f}")
            plots[0][i].legend()
            plots[0][i].grid()
        fig.suptitle('Нормалізовані графіки', fontsize=16, y=1.05)
        return fig

    def plot_predict(self):
        data = range(len(self.model.y))

        fig, plots = plot.subplots(1, self.model.ny, squeeze=False, figsize=(15, 4))
        for i in range(self.model.ny):
            plots[0][i].plot(data, self.model.y[:, i], label=f'Y{i+1}')
            plots[0][i].plot(data, self.model.predict[:, i], label=f'Ф{i+1}')
            plots[0][i].set_title(f"Похибка: {self.model.error[i]:.4f}")
            plots[0][i].legend()
            plots[0][i].grid()
        fig.suptitle('Відновлені графіки', fontsize=16, y=1.05)
        return fig