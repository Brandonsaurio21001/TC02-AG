from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict
from .poblacion import Poblacion
from .individuo import Individuo
from .fitness import evaluar_fitness

class GeneticAlgorithm:
    def __init__(self, config: Dict):
        self.cfg = config
        assert self.cfg["population_size"] >= 30, "Requisito: población >= 30"
        assert self.cfg["generations"] >= 50, "Requisito: generaciones >= 50"
        self.poblacion = Poblacion(tamano=self.cfg["population_size"], num_inputs=4)

    # -------- selección --------
    def _torneo(self, k: int = 3) -> Individuo:
        candidatos = np.random.choice(self.poblacion.individuos, size=k, replace=False)
        return max(candidatos, key=lambda ind: ind.fitness)

    # -------- cruce (uniforme) --------
    def _cruzar(self, p1: Individuo, p2: Individuo) -> Individuo:
        mask = np.random.rand(p1.pesos.size) < 0.5
        child_w = np.where(mask, p1.pesos, p2.pesos)
        hijo = Individuo(num_inputs=p1.num_inputs, pesos=child_w.copy())
        return hijo

    def _evaluar_poblacion(self) -> None:
        for ind in self.poblacion.individuos:
            ind.fitness = evaluar_fitness(
                ind,
                env_name=self.cfg.get("env_name", "CartPole-v1"),
                episodios=self.cfg.get("episodes_per_eval", 3),
                k_pos=self.cfg.get("k_pos", 0.5),
                k_ang=self.cfg.get("k_ang", 1.0),
            )

    def run(self) -> Dict[str, List[float]]:
        historia_max, historia_avg = [], []

        # evaluar generación 0
        self._evaluar_poblacion()
        historia_max.append(self.poblacion.mejor().fitness)
        historia_avg.append(self.poblacion.promedio_fitness())

        for gen in range(1, self.cfg["generations"] + 1):
            nueva: List[Individuo] = []

            # elitismo simple
            elite_n = self.cfg.get("elitism", 1)
            elites = sorted(self.poblacion.individuos, key=lambda i: i.fitness, reverse=True)[:elite_n]
            nueva.extend([Individuo(num_inputs=e.num_inputs, pesos=e.pesos.copy()) for e in elites])

            # reproducción
            while len(nueva) < self.poblacion.tamano:
                p1, p2 = self._torneo(self.cfg.get("tournament_k", 3)), self._torneo(self.cfg.get("tournament_k", 3))
                if np.random.rand() < self.cfg.get("crossover_rate", 0.9):
                    hijo = self._cruzar(p1, p2)
                else:
                    hijo = Individuo(num_inputs=p1.num_inputs, pesos=p1.pesos.copy())

                hijo.mutar(self.cfg.get("mutation_rate", 0.05), sigma=self.cfg.get("mutation_sigma", 0.1))
                nueva.append(hijo)

            self.poblacion.individuos = nueva[:self.poblacion.tamano]
            self._evaluar_poblacion()

            historia_max.append(self.poblacion.mejor().fitness)
            historia_avg.append(self.poblacion.promedio_fitness())

            if self.cfg.get("verbose", True) and gen % 5 == 0:
                print(f"Gen {gen:03d} | max={historia_max[-1]:.2f} | avg={historia_avg[-1]:.2f}")

        return {"max": historia_max, "avg": historia_avg, "best": self.poblacion.mejor()}
