import sys
import numpy as np
import gymnasium as gym
from src.individuo import Individuo

def ver_jugar(path_pesos: str, env_name: str = "CartPole-v1"):
    # 1. Cargar pesos del archivo
    with open(path_pesos, "r") as f:
        pesos = np.array([float(x) for x in f.read().strip().split()])

    # 2. Crear individuo con esos pesos
    ind = Individuo(num_inputs=4, pesos=pesos)

    # 3. Crear entorno con render activado
    env = gym.make(env_name, render_mode="human")

    obs, _ = env.reset()
    done, trunc = False, False
    pasos = 0

    while not (done or trunc):
        accion = ind.decidir(obs)            # Usar política aprendida
        obs, recompensa, done, trunc, _ = env.step(accion)
        pasos += 1

    env.close()
    print(f"El individuo jugó {pasos} pasos.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m src.watch_best <ruta_best_weights.txt>")
        sys.exit(1)

    ver_jugar(sys.argv[1])
