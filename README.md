# TC02-AG

### Ejecución del proyecto 
1. Teniendo instalado python se realiza el comando:
```python
py -m pip install "gymnasium[classic-control]" numpy matplotlib  
```
2. Descargando el zip, en la carpeta principal (TC02-AG), se ejecuta alguna configuración con el comando:
```python
py -m src.main experiments/[archivo.json].
```
Por ejemplo: 
```python
py -m src.main experiments/config1.json
```

### Ejecución del mejor individuo
Para poder al mejor individuo resultado de una ejecución se utiliza el comando: 

```python
py -m src.watch_best results/[carpeta en results]/best_weights.txt
```

Un ejemplo seria: 

```python
python -m src.watch_best results/config1_20250920_215917/best_weights.txt
```
