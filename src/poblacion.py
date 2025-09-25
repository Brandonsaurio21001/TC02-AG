from __future__ import annotations
import numpy as np
from typing import List
from .individuo import Individuo

class Poblacion:
    # Se inicializa la poblacion con una lista de individos de pesos aleatorios de un tamaño pasado.
    def __init__(self, tamano: int, num_inputs: int = 4):
        self.tamano = tamano
        self.individuos: List[Individuo] = [Individuo.aleatorio(num_inputs) for _ in range(tamano)]

    # Se retorna el individuo con mayor fitness
    def mejor(self) -> Individuo:
        return max(self.individuos, key=lambda ind: ind.fitness)

    # Se retorna el promedio de fitness en la poblacion. 
    def promedio_fitness(self) -> float:
        return float(np.mean([ind.fitness for ind in self.individuos]))
