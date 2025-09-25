from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict
from .poblacion import Poblacion
from .individuo import Individuo
from .fitness import evaluar_fitness

class GeneticAlgorithm:
    def __init__(self, config: Dict):
        self.cfg = config
        #assert self.cfg["population_size"] >= 30, "Requisito: población >= 30"
        #assert self.cfg["generations"] >= 50, "Requisito: generaciones >= 50"
        self.poblacion = Poblacion(tamano=self.cfg["population_size"], num_inputs=4)

        # Render persistente (opcional)
        self._render_env = None
        self._render_persistent = self.cfg.get("render_persistent", False)

    # Dado un numero k, se seleccionan aleatoriamente k individuos de la poblacion y se retorna el individuo con mayor fitness.
    def _torneo(self, k: int = 3) -> Individuo:
        candidatos = np.random.choice(self.poblacion.individuos, size=k, replace=False)
        return max(candidatos, key=lambda ind: ind.fitness)

    # Dado dos individios se cruza de manera uniforme los pesos para generar un hijo que los herede.
    def _cruzar(self, p1: Individuo, p2: Individuo) -> Individuo:
        mask = np.random.rand(p1.pesos.size) < 0.5
        child_w = np.where(mask, p1.pesos, p2.pesos)
        return Individuo(num_inputs=p1.num_inputs, pesos=child_w.copy())

    # Evalua el fitness de cada individuo de la poblacion.
    def _evaluar_poblacion(self) -> None:
        for ind in self.poblacion.individuos:
            ind.fitness = evaluar_fitness(
                ind,
                env_name=self.cfg.get("env_name", "CartPole-v1"),
                episodios=self.cfg.get("episodes_per_eval", 3),
                k_pos=self.cfg.get("k_pos", 0.5),
                k_ang=self.cfg.get("k_ang", 1.0),
            )
    # Muestra en una ventana cómo se comporta el mejor cada render_every generaciones, hasta render_max_steps.
    def _render_best(self, gen: int) -> None:
        if not self.cfg.get("render_env", False): return
        if gen % self.cfg.get("render_every", 10) != 0: return

        import gymnasium as gym
        best = self.poblacion.mejor()
        max_steps = int(self.cfg.get("render_max_steps", 500))

        if self._render_persistent:
            if self._render_env is None:
                self._render_env = gym.make(self.cfg.get("env_name", "CartPole-v1"), render_mode="human")
            env = self._render_env
        else:
            env = gym.make(self.cfg.get("env_name", "CartPole-v1"), render_mode="human")

        obs, _ = env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc) and steps < max_steps:
            action = best.decidir(obs)
            obs, _, done, trunc, _ = env.step(action)
            steps += 1

        if not self._render_persistent:
            env.close()

    
    def _checkpoint_best(self, gen: int) -> None:
        ckpt_every = int(self.cfg.get("checkpoint_every", 0))
        outdir = self.cfg.get("outdir", None)
        if ckpt_every <= 0 or not outdir: return
        if gen % ckpt_every != 0: return
        best = self.poblacion.mejor()
        path = f'{outdir}/gen_{gen:03d}_weights.txt'
        with open(path, "w", encoding="utf-8") as f:
            f.write(" ".join([f"{w:.6f}" for w in best.pesos]))

    def run(self) -> Dict[str, List[float]]:
        historia_max, historia_avg = [], []

        self._evaluar_poblacion()
        historia_max.append(self.poblacion.mejor().fitness)
        historia_avg.append(self.poblacion.promedio_fitness())

        if self.cfg.get("verbose", True):
            print(f"Gen 000 | max={historia_max[-1]:.2f} | avg={historia_avg[-1]:.2f}")

        self._render_best(0)
        self._checkpoint_best(0)

        for gen in range(1, self.cfg["generations"] + 1):
            nueva: List[Individuo] = []

            elite_n = self.cfg.get("elitism", 1)
            elites = sorted(self.poblacion.individuos, key=lambda i: i.fitness, reverse=True)[:elite_n]
            nueva.extend([Individuo(num_inputs=e.num_inputs, pesos=e.pesos.copy()) for e in elites])

            while len(nueva) < self.poblacion.tamano:
                p1 = self._torneo(self.cfg.get("tournament_k", 3))
                p2 = self._torneo(self.cfg.get("tournament_k", 3))
                if np.random.rand() < self.cfg.get("crossover_rate", 0.9):
                    hijo = self._cruzar(p1, p2)
                else:
                    hijo = Individuo(num_inputs=p1.num_inputs, pesos=p1.pesos.copy())

                hijo.mutar(
                    self.cfg.get("mutation_rate", 0.05),
                    sigma=self.cfg.get("mutation_sigma", 0.1)
                )
                nueva.append(hijo)

            self.poblacion.individuos = nueva[:self.poblacion.tamano]
            self._evaluar_poblacion()

            historia_max.append(self.poblacion.mejor().fitness)
            historia_avg.append(self.poblacion.promedio_fitness())

            if self.cfg.get("verbose", True):# and gen % 5 == 0:
                print(f"Gen {gen:03d} | max={historia_max[-1]:.2f} | avg={historia_avg[-1]:.2f}")

            self._render_best(gen)
            self._checkpoint_best(gen)

        if self._render_env is not None:
            self._render_env.close()

        return {"max": historia_max, "avg": historia_avg, "best": self.poblacion.mejor()}
