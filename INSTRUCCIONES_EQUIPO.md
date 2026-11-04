# TP3: instrucciones para las Partes 2 y 3

La Parte 1 está terminada y subida. Faltan la **Parte 2** (análisis de errores y espacio latente) y la **Parte 3** (transfer learning). Entrega: **viernes 13 de noviembre, 23:59**, con el código y un pequeño informe explicativo.

No hace falta saber mucho de CNNs para seguir esta guía: los conceptos están explicados y el código ya se probó. Lo que más suma nota es **interpretar los resultados con números**.

## 1. Qué hicimos en la Parte 1

**El problema.** EuroSAT tiene 27.000 imágenes satelitales RGB de 64 × 64 píxeles, y hay que clasificar cada una en 10 tipos de uso del suelo, como Forest, River o Highway. Están divididas en 18.900 de train, 5.400 de validation y 2.700 de test.

**Preparación de los datos.**
- Normalizamos cada canal de color con la media y el desvío de **train**. No usamos validation ni test para calcularlos, porque sería data leakage.
- Entrenamos por batches de 64 imágenes.
- Aplicamos data augmentation: en cada batch, las imágenes de train se espejan y se rotan de a 90° al azar. Una foto satelital no tiene "arriba", así que rotarla no cambia su clase. Esto hace que la red vea más variedad y reduce el riesgo de overfitting.

**Qué es una CNN.** Es una red que aprende filtros: pequeñas ventanas que recorren la imagen buscando patrones.
- **Capas convolucionales:** detectan patrones, primero simples (bordes, colores) y después más complejos (texturas, formas).
- **MaxPool:** achica la imagen a la mitad, para que las capas siguientes miren una zona más grande.
- **Capas lineales:** toman lo detectado y dan un puntaje por clase. Gana la clase con el puntaje más alto.

**Nuestra red (`red_v2`).** Tiene 3 bloques convolucionales y 2 capas lineales, con 1.143.242 parámetros.

**Entrenamiento.** Entrenamos 25 épocas con Adam (`lr=0.001`) y nos quedamos con los pesos de la época con mejor accuracy en validation. Esos pesos están guardados en `red_v2.pt`.

| Resultado de `red_v2` | Valor |
|---|---|
| Mejor época | 22 |
| Accuracy / loss en validation | 0.952 / 0.142 |
| Primera época con accuracy de validation ≥ 0.90 | 7 |
| Primera época con accuracy de validation ≥ 0.93 | 10 |
| Primera época con accuracy de validation ≥ 0.95 | 22 |
| Tiempo por época (CPU, laptop de Manuel) | 104.6 ± 8.5 s, sin contar 2 épocas anómalas (867 s y 9898 s) |

**Gradientes.** Medimos si el aprendizaje llega a todas las capas durante las primeras 3 épocas, y no encontramos vanishing ni exploding.

## 2. Cómo arrancar sin reentrenar

1. Hacé `git pull` para bajarte el notebook y `red_v2.pt`.
2. Instalá lo necesario: `pip install torch torchvision pandas matplotlib seaborn scikit-learn`
3. Abrí `tp3.ipynb`.

> **No uses "Run All".** Volvería a entrenar las dos redes: más de una hora en CPU. **Tampoco corras la celda con `torch.save(red_v2...)` antes de cargar los pesos**, porque pisaría el modelo guardado con una red sin entrenar.

Corré solo estas celdas, en este orden:

1. Bibliotecas (`# Bibliotecas`)
2. Lectura de los CSV (`train_df = pd.read_csv...`)
3. Limpieza (`# Limpieza de datos:`)
4. Tensores de train (`# Leemos cada imagen de train...`)
5. Media y desvío (`# Escalamos a [0, 1]...`)
6. `def cargar_split(df)`
7. `BATCH_SIZE = 64`
8. `def aumentar(x)`
9. `red_v2 = nn.Sequential(...)`
10. `def evaluar(...)`, que también define `entrenar`
11. `def entrenar_v2(...)`

Después agregá al final esta celda, que carga el modelo y comprueba que quedó bien:

```python
red_v2.load_state_dict(torch.load("red_v2.pt"))
loss_val, acc_val = evaluar(red_v2, val_loader, nn.CrossEntropyLoss())
print(f"Validation: loss {loss_val:.3f}, accuracy {acc_val:.3f}")
```

**Tiene que dar loss 0.142 y accuracy 0.952.** Si da otra cosa, algo se cargó mal: frená y revisá antes de seguir.

## 3. Reglas de trabajo

El corrector es muy exigente y en los TPs anteriores nos bajó puntos por estas cosas.

**Prolijidad**
- **No borres ni reemplaces celdas:** si algo cambia, agregá otra versión debajo con un nombre nuevo (como `red` y `red_v2`). Queremos que se vea el camino que hicimos.
- **Título y explicación:** cada paso lleva una explicación en markdown, con el mismo formato (`> texto`).
- **Secciones:** la Parte 2 va bajo el título `#### 3- Análisis de Errores y Espacio Latente.`, que ya está en el notebook. La Parte 3 va bajo `#### 4- Transfer Learning.`
- **Código simple:** tiene que poder explicarlo cualquiera del grupo. Usen lo visto en clase, y si algo es nuevo, expliquen en un comentario qué hace.

**Interpretación**
- **Todo resultado se interpreta:** un gráfico o una tabla sin texto no alcanza.
- **Citá números concretos:** no escribas "confunde bastante", sino "confunde 12 de 200 imágenes (6%)".
- **Verificá lo que afirmás:** si no lo comprobaste con código, presentalo como hipótesis ("puede deberse a...").
- **Mostrá dispersión:** los promedios van con su desvío.
- **Todo lo importante va al informe:** nada debería quedar solo en el notebook.

**Experimentos**
- **Test se usa una sola vez por modelo:** las decisiones (épocas, learning rate) se toman con validation. Si elegís algo mirando test, el resultado deja de ser una estimación honesta.
- **Reproducibilidad:** poné `torch.manual_seed(0)` antes de crear una red y antes de entrenarla.
- **Reentrenar exige recrear la red:** si no, sigue entrenando desde los pesos anteriores.

**Git**
- Nunca editen `tp3.ipynb` dos personas a la vez: los conflictos en notebooks son muy difíciles de resolver.
- Trabajen por turnos: `git pull`, trabajar, guardar, commit, `git push` y avisar al grupo.
- Si necesitan avanzar en paralelo, usen una copia cada uno (`parte2.ipynb`, `parte3.ipynb`) y al final pasen las celdas a `tp3.ipynb`, comprobando que se mantengan las salidas.

## 4. Parte 2: análisis de errores y espacio latente

### Paso 1: métricas en test

Esta es la primera vez que el modelo ve test, así que esta nota es la estimación honesta de su rendimiento.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Nombres de las clases, ordenados según su Label (0 a 9)
clases = train_df.groupby("Label")["ClassName"].first().tolist()

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

def predecir(red, loader):
    """Devuelve la clase predicha para cada imagen del loader."""
    red.eval()
    predicciones = []
    with torch.no_grad():
        for x, y in loader:
            predicciones.append(red(x).argmax(dim=1))
    return torch.cat(predicciones)

pred_custom = predecir(red_v2, test_loader)
print(classification_report(y_test, pred_custom, target_names=clases, digits=3))
```

Qué significa cada métrica:
- **Accuracy:** de todas las imágenes, qué proporción acertó.
- **Precision de una clase:** de las que la red dijo que son de esa clase, cuántas lo eran.
- **Recall de una clase:** de las que eran de esa clase, cuántas encontró.
- **F1:** combina precision y recall en un solo número.

Qué escribir: el accuracy y el F1 macro de test, las clases mejor y peor clasificadas, y cómo le fue a **Pasture**, la clase con menos imágenes. Compará también con el 0.952 de validation. Es esperable que test dé algo distinto, porque la mejor época se eligió mirando validation.

### Paso 2: matriz de confusión

Cada fila es la clase real y cada columna, la predicha. La diagonal son los aciertos, y todo lo demás son errores.

```python
matriz = confusion_matrix(y_test, pred_custom)

fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases, ax=ax)
ax.set_title("Matriz de confusión de red_v2 en test")
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
plt.show()

# Ranking de las confusiones más frecuentes
confusiones = []
for i, real in enumerate(clases):
    for j, predicha in enumerate(clases):
        if i != j and matriz[i, j] > 0:
            confusiones.append({"Real": real, "Predicha": predicha, "Errores": matriz[i, j],
                                "% de la clase real": round(100 * matriz[i, j] / matriz[i].sum(), 1)})
print(pd.DataFrame(confusiones).sort_values("Errores", ascending=False).head(10))
```

Miren también el porcentaje, porque las clases tienen distinta cantidad de imágenes en test (entre 200 y 300). Además, "A confundida con B" no es lo mismo que "B confundida con A": comenten cada dirección.

### Paso 3: justificar las confusiones

La consigna pide **justificar** los errores, no solo listarlos. Para cada una de las 3 o 4 confusiones principales, miren imágenes mal clasificadas:

```python
# Cambiá estas dos clases según lo que muestre tu ranking
real, predicha = clases.index("Pasture"), clases.index("HerbaceousVegetation")

indices = ((y_test == real) & (pred_custom == predicha)).nonzero().flatten()[:5]
print(f"Se muestran {len(indices)} errores")

if len(indices) > 0:
    # Deshacemos la normalización para ver los colores reales
    imagenes = X_test[indices] * desvio.view(1, 3, 1, 1) + media.view(1, 3, 1, 1)
    imagenes = imagenes.permute(0, 2, 3, 1).clamp(0, 1)

    fig, axs = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3), constrained_layout=True, squeeze=False)
    for ax, imagen in zip(axs[0], imagenes):
        ax.imshow(imagen)
        ax.axis("off")
    plt.suptitle(f"Real: {clases[real]}, predicha: {clases[predicha]}")
    plt.show()
```

Usen el EDA como punto de partida, pero comprueben si la matriz lo confirma. Ahí anticipamos pares parecidos (Pasture y HerbaceousVegetation, AnnualCrop y PermanentCrop, Highway y River, Industrial y Residential) y vimos con boxplots que algunos tienen colores muy superpuestos. Un buen argumento une los tres datos: cuántos errores hay, qué se ve en las imágenes y qué decían los colores.

Si queda tiempo, la clase 18 explica oclusión, saliency maps y Grad-CAM, que muestran en qué zona de la imagen se fijó la red. Es un extra, no algo obligatorio.

### Paso 4: espacio latente

Justo antes de decidir, la red resume cada imagen en un vector de 128 números. Es su "representación interna", y a eso se le llama espacio latente. Si las imágenes de una misma clase quedan juntas ahí, a la última capa le resulta fácil separarlas.

No podemos graficar 128 dimensiones, así que las proyectamos a 2:
- **PCA:** busca las direcciones con más variación. Es lineal y conserva la estructura global.
- **t-SNE:** intenta mantener juntas las imágenes que eran vecinas. Sirve mucho para ver grupos.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# red_v2 sin su última capa: la salida es el vector de 128 números
extractor = red_v2[:-1]
extractor.eval()

features = []
with torch.no_grad():
    for x, y in test_loader:
        features.append(extractor(x))
features = torch.cat(features).numpy()
print(f"Forma de los features: {features.shape}")  # (2700, 128)

# PCA a 50 dimensiones antes de t-SNE, para quitar ruido y acelerar
pca = PCA(n_components=50, random_state=0)
features_pca = pca.fit_transform(features)
print(f"Varianza explicada con 2 componentes: {pca.explained_variance_ratio_[:2].sum():.3f}")
print(f"Varianza explicada con 50 componentes: {pca.explained_variance_ratio_.sum():.3f}")

coords_tsne = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(features_pca)

fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
for i, clase in enumerate(clases):
    filtro = y_test == i
    axs[0].scatter(features_pca[filtro, 0], features_pca[filtro, 1], s=8, alpha=0.7, label=clase, color=plt.cm.tab10(i))
    axs[1].scatter(coords_tsne[filtro, 0], coords_tsne[filtro, 1], s=8, alpha=0.7, label=clase, color=plt.cm.tab10(i))
axs[0].set_title("PCA del espacio latente (test)")
axs[1].set_title("t-SNE del espacio latente (test)")
axs[1].legend(markerscale=2)
plt.show()
```

Qué analizar:
- ¿Qué clases forman grupos bien separados?
- ¿Cuáles se mezclan? ¿Son las mismas que se confunden en la matriz? Esa relación es lo más interesante para contar.

**Un cuidado con t-SNE:** el tamaño de un grupo y la distancia entre grupos lejanos no tienen un significado confiable. Lo que sí se puede leer es qué puntos quedan juntos o mezclados. No concluyan "River está lejos de Forest, así que son muy distintas" solo por el t-SNE.

## 5. Parte 3: transfer learning

**La idea.** En vez de empezar de cero, tomamos una red ya entrenada con ImageNet, un dataset de más de un millón de fotos. Sus filtros ya detectan bordes y texturas, algo útil también en imágenes satelitales. Reemplazamos la última capa, porque ImageNet tiene 1000 clases y nosotros 10, y reentrenamos solo el final de la red.

**Por qué ResNet18.** Es mucho más liviana que VGG16 (unos 11 millones de parámetros contra unos 138 millones), algo importante porque entrenamos en CPU.

### Paso 1: normalizar como espera ResNet

ResNet18 aprendió con imágenes normalizadas con la media y el desvío de ImageNet, no con los de EuroSAT. Por eso, deshacemos nuestra normalización y aplicamos la de ImageNet.

```python
from torchvision import models

media_imagenet = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
desvio_imagenet = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def a_imagenet(X):
    """Deshace la normalización de EuroSAT y aplica la de ImageNet."""
    X_original = X * desvio.view(1, 3, 1, 1) + media.view(1, 3, 1, 1)
    return (X_original - media_imagenet) / desvio_imagenet

train_loader_tl = DataLoader(TensorDataset(a_imagenet(X_train), y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader_tl = DataLoader(TensorDataset(a_imagenet(X_val), y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader_tl = DataLoader(TensorDataset(a_imagenet(X_test), y_test), batch_size=BATCH_SIZE, shuffle=False)
```

Dejamos las imágenes en 64 × 64, igual que en la red propia, aunque ResNet18 se entrenó con imágenes más grandes (224 × 224). Agrandarlas podría mejorar el resultado, pero sería mucho más lento en CPU. Mencionen esta limitación en el informe.

### Paso 2: adaptar la red

"Congelar" una capa es no actualizar sus pesos. Congelamos casi todo, y entrenamos solo el último bloque (`layer4`) y la capa final nueva (`fc`).

```python
torch.manual_seed(0)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # la primera vez descarga 45 MB

# Congelamos toda la red
for parametro in resnet.parameters():
    parametro.requires_grad = False

# Descongelamos el último bloque
for parametro in resnet.layer4.parameters():
    parametro.requires_grad = True

# Reemplazamos la capa final: 10 clases en lugar de 1000
resnet.fc = nn.Linear(512, 10)

entrenables = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"Parámetros totales: {sum(p.numel() for p in resnet.parameters()):,}; entrenables: {entrenables:,}")
# Debería imprimir 11,181,642 totales y 8,398,858 entrenables
```

### Paso 3: entrenar

**No se puede reutilizar `entrenar_v2`:** recorre la red capa por capa, algo que funciona con `nn.Sequential` pero da error con ResNet, y además usa los loaders con la normalización de EuroSAT. Esta versión es igual, sin el registro de gradientes y recibiendo los loaders como parámetros.

```python
def entrenar_tl(red, n_epocas, train_loader, val_loader, lr=0.0001):
    """Igual que entrenar_v2, sin registrar gradientes y recibiendo los loaders."""
    optimizer = torch.optim.Adam([p for p in red.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    historia = {"loss_train": [], "acc_train": [], "loss_val": [], "acc_val": [], "tiempo": []}
    mejor_acc, mejor_epoca, mejor_estado = 0, 0, None

    for epoca in range(n_epocas):
        inicio = time.time()
        red.train()
        loss_total, aciertos, n = 0, 0, 0

        for x, y in train_loader:
            x = aumentar(x)
            optimizer.zero_grad()
            salida = red(x)
            loss = criterion(salida, y)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * len(y)
            aciertos += (salida.argmax(dim=1) == y).sum().item()
            n += len(y)

        historia["loss_train"].append(loss_total / n)
        historia["acc_train"].append(aciertos / n)
        loss_val, acc_val = evaluar(red, val_loader, criterion)
        historia["loss_val"].append(loss_val)
        historia["acc_val"].append(acc_val)
        historia["tiempo"].append(time.time() - inicio)

        if acc_val > mejor_acc:
            mejor_acc, mejor_epoca = acc_val, epoca + 1
            mejor_estado = copy.deepcopy(red.state_dict())

        print(f"Época {epoca + 1:2d}/{n_epocas} | "
              f"loss train {historia['loss_train'][-1]:.3f}, val {loss_val:.3f} | "
              f"acc train {historia['acc_train'][-1]:.3f}, val {acc_val:.3f} | "
              f"{historia['tiempo'][-1]:.0f} s")

    red.load_state_dict(mejor_estado)
    print(f"\nMejor época: {mejor_epoca} (accuracy validation = {mejor_acc:.3f})")
    historia["mejor_epoca"] = mejor_epoca
    return historia

N_EPOCAS_TL = 15

torch.manual_seed(0)
historia_tl = entrenar_tl(resnet, N_EPOCAS_TL, train_loader_tl, val_loader_tl)

# Guardamos los pesos para no tener que reentrenar
torch.save(resnet.state_dict(), "resnet18_tl.pt")
```

**Referencia:** en una prueba de una sola época en la laptop de Manuel, tardó 190 s y llegó a accuracy de validation 0.932. Sirve para saber si van bien, no como resultado para el informe. A ese ritmo, 15 épocas son unos 48 minutos. Dejen la compu enchufada, sin suspenderse y sin otros programas pesados, porque si no los tiempos salen inservibles (nos pasó en la Parte 1).

Decisiones que hay que justificar:
- **Learning rate 0.0001, menor que el de la red propia:** los pesos ya son buenos y queremos ajustarlos de a poco.
- **Cantidad de épocas:** 15 es un punto de partida. Graficá loss y accuracy igual que en la Parte 1 y decidí mirando las curvas. Si cambiás algo, hacelo en una celda nueva.
- **Modo train en capas congeladas:** las capas BatchNorm de ResNet actualizan sus promedios internos durante `red.train()`, incluso congeladas. No hace falta cambiar nada, pero conviene saberlo si lo preguntan.

### Paso 4: evaluar en test

Reutilizá `predecir` y el código de la matriz de confusión de la Parte 2, ahora con `resnet` y `test_loader_tl`. Comparen también qué confusiones mejoraron y cuáles siguen.

### Paso 5: comparar con la red propia

La consigna pide tres comparaciones.

**1. Velocidad de convergencia.** ¿En qué época llega cada red a cierto accuracy de validation? Los valores de `red_v2` están en la tabla de la sección 1. Conviene sumar un gráfico con las dos curvas de accuracy de validation.

**2. Rendimiento final.** Accuracy, F1 macro y métricas por clase, siempre en test.

**3. Tiempo por época.** **Tienen que medir las dos redes en la misma computadora**: los 104.6 s de la tabla son de la laptop de Manuel y no se pueden comparar con otra máquina. Si ResNet se entrena en otra compu, corran esto ahí:

```python
# Solo para medir tiempos: entrena una copia, así red_v2 no cambia
historia_tiempo, _ = entrenar_v2(copy.deepcopy(red_v2), 5)
tiempos = torch.tensor(historia_tiempo["tiempo"])
print(f"Red propia: {tiempos.mean():.1f} ± {tiempos.std():.1f} s por época")

tiempos_tl = torch.tensor(historia_tl["tiempo"])
print(f"ResNet18: {tiempos_tl.mean():.1f} ± {tiempos_tl.std():.1f} s por época")
```

Si alguna época tarda muchísimo más que el resto, repórtenla y aclaren que la excluyen. No la borren en silencio.

Resuman todo en una tabla, completada con sus resultados:

| Criterio | Red propia | ResNet18 |
|---|---|---|
| Parámetros totales / entrenables | 1.143.242 / 1.143.242 | 11.181.642 / 8.398.858 |
| Primera época con accuracy de validation ≥ 0.90 | 7 | |
| Primera época con accuracy de validation ≥ 0.95 | 22 | |
| Mejor accuracy de validation (época) | 0.952 (22) | |
| Accuracy en test | | |
| F1 macro en test | | |
| Tiempo por época (misma compu, media ± desvío) | | |

Cierren con una conclusión: qué se gana y qué se paga con cada enfoque. Por ejemplo, si ResNet tarda más por época pero necesita menos épocas, comparen también el **tiempo total** hasta llegar a un mismo accuracy.

## 6. Antes de entregar

- [ ] Cada gráfico y tabla tiene su interpretación con números.
- [ ] La matriz de confusión es de **test**, no de train ni de validation.
- [ ] Las confusiones principales están justificadas, con cantidades de errores e imágenes.
- [ ] El espacio latente responde si las clases forman clusters y lo relaciona con la matriz.
- [ ] La comparación cubre convergencia, métricas y tiempo por época, medidos en la misma computadora.
- [ ] Los resultados están guardados en el notebook, sin celdas con errores.
- [ ] El informe cubre las tres partes, incluida la Parte 1.
- [ ] Definimos quién escribe el informe y quién lo sube al Campus (solo sube **una** persona).
- [ ] Sacamos este archivo de la entrega.

Si no entendés un resultado, no lo escribas como conclusión hasta entenderlo: cualquiera del grupo tiene que poder explicarlo.
