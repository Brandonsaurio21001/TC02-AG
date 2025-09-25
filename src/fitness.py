from __future__ import annotations
import numpy as np
import gymnasium as gym
from .individuo import Individuo

def evaluar_fitness(individuo: Individuo, env_name: str = "CartPole-v1", episodios: int = 3,
                    k_pos: float = 0.5, k_ang: float = 1.0) -> float:
    """
    Fitness con penalización:
      fitness_ep = sum(recompensa - (k_pos*|x| + k_ang*|theta|)) por paso.
    Promediado en varios episodios para reducir varianza.
    """
    env = gym.make(env_name)
    total = 0.0
    for _ in range(episodios):
        obs, _ = env.reset()
        done = False
        ep = 0.0
        while not done:
            accion = individuo.decidir(obs)
            obs, r, done, trunc, _ = env.step(accion)
            x, x_dot, theta, theta_dot = obs
            penal = k_pos * abs(x) + k_ang * abs(theta)
            ep += (r - penal) # Para penalizar.
            #ep += r # Si no quiero penalizar.
            if trunc:  # Por si el entorno trunca al alcanzar el límite.
                break
        total += ep
    env.close()
    return total / episodios
