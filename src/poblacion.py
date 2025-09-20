from __future__ import annotations
import numpy as np
from typing import List
from .individuo import Individuo

class Poblacion:
    def __init__(self, tamano: int, num_inputs: int = 4):
        self.tamano = tamano
        self.individuos: List[Individuo] = [Individuo.aleatorio(num_inputs) for _ in range(tamano)]

    def mejor(self) -> Individuo:
        return max(self.individuos, key=lambda ind: ind.fitness)

    def promedio_fitness(self) -> float:
        return float(np.mean([ind.fitness for ind in self.individuos]))
