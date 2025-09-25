import json, os
import matplotlib.pyplot as plt
from datetime import datetime
from .ga import GeneticAlgorithm

def plot_historia(hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    gens = list(range(len(hist["max"])))
    plt.figure()
    plt.plot(gens, hist["avg"], label="Fitness promedio")
    plt.plot(gens, hist["max"], label="Fitness máximo")
    plt.xlabel("Generación"); plt.ylabel("Fitness")
    plt.title("Evolución del fitness")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fitness.png"))
    plt.close()

def run_with_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Crear carpeta de resultados y pasarla a cfg
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"{os.path.basename(config_path).split('.')[0]}_{stamp}")
    cfg["outdir"] = outdir
    os.makedirs(outdir, exist_ok=True)  

    ga = GeneticAlgorithm(cfg)
    hist = ga.run()

    plot_historia(hist, outdir)

    # Guardar pesos del mejor
    best = hist["best"]
    with open(os.path.join(outdir, "best_weights.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join([f"{w:.6f}" for w in best.pesos]))
    print(f"Resultados guardados en {outdir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python -m src.main experiments/configX.json")
        sys.exit(1)
    run_with_config(sys.argv[1])
