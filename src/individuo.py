from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Individuo:
    """Cromosoma = pesos lineales + sesgo para una política binaria."""
    num_inputs: int = 4
    pesos: np.ndarray = field(default_factory=lambda: np.zeros(5))  # 4 pesos + 1 bias
    fitness: float = -np.inf

    @staticmethod
    def aleatorio(num_inputs: int = 4, escala: float = 0.5) -> "Individuo":
        # 4 pesos + 1 bias
        w = np.random.uniform(-escala, escala, size=num_inputs + 1)
        return Individuo(num_inputs=num_inputs, pesos=w)

    def decidir(self, obs: np.ndarray) -> int:
        """Política lineal: acción = 1 si w·x + b >= 0, si no 0."""
        x = np.append(obs, 1.0)               # añadir bias
        logits = float(np.dot(self.pesos, x))
        return 1 if logits >= 0 else 0

    def mutar(self, tasa_mutacion: float, sigma: float = 0.1) -> None:
        """Mutación gaussiana independiente por gen."""
        mascara = np.random.rand(self.pesos.size) < tasa_mutacion
        self.pesos = self.pesos + mascara * np.random.normal(0.0, sigma, size=self.pesos.size)
