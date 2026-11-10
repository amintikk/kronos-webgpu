<img src="assets/icon.svg" width="64" height="64" alt="">

# Kronos WebGPU

Predicción de velas (criptomonedas u otro OHLCV) con modelos de series temporales que se ejecutan
**en el navegador**: WebGPU si está disponible y WebAssembly con hilos si no. No hay backend: la web
es estática, los pesos se descargan una vez a la caché del navegador y las velas se piden
directamente a la bolsa.

**Pruébalo:** https://kronos.aminox.dev (página de presentación) · https://kronos.aminox.dev/app/ (la app)

> No es asesoramiento financiero. Los modelos dan escenarios estadísticos que a menudo fallan; la app
> incluye backtest precisamente para comprobarlo.

## Estructura del sitio

| Ruta | Archivo | Contenido |
|---|---|---|
| `/` | `index.html`, `src/site/` | Presentación del proyecto con una predicción de ejemplo (`public/demo.json`) |
| `/app/` | `app/index.html`, `src/main.ts` | La herramienta |
| `/aviso-legal/`, `/privacidad/`, `/terminos/`, `/licencias/` | `*/index.html` | Textos legales |

Las páginas estáticas comparten cabecera y pie (`src/site/partials/`), que Vite inserta al compilar.
`public/demo.json` se regenera con `KRONOS_MODELS=/ruta/a/modelos node tools/make_demo.ts > public/demo.json`.

## Qué incluye

- **Modelos**: [Kronos](https://github.com/shiyu-coder/Kronos) mini / small / base (MIT) y
  [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) tiny / small (Apache-2.0).
- **Datos**: Binance (con websocket en directo), Bybit, OKX, Coinbase y CSV propio.
- **Predicción**: horizonte, número de trayectorias, temperatura, top-p, top-k, semilla, punto de
  inicio en el pasado, predicción automática al cerrar cada vela.
- **Gráfico propio en canvas** (`src/chart.ts`, sin librerías): velas reales y previstas en un mismo eje
  de tiempo que continúa tras la última vela, abanico de percentiles, trayectorias, mediana con
  animación de aparición, volumen, escala logarítmica y tema claro y oscuro. Arrastrar con inercia,
  rueda o pellizco para ampliar, doble clic para volver a la predicción y ficha flotante con la vela y
  los percentiles de cada paso. Se pueden fijar hasta dos predicciones para compararlas con otra.
- **Evaluación**: comparación con lo ocurrido y backtest por ventanas (MAPE, habilidad frente a
  «el precio no se mueve», dirección, cobertura de la banda).
- **Salida**: CSV, JSON, PNG y enlace compartible con la configuración.

## Cómo funciona la inferencia

El bucle de Kronos original recalcula todo el contexto en cada vela. `tools/export_kronos.py`
reimplementa el modelo con los mismos pesos y una caché de claves/valores, y lo exporta en cuatro
grafos ONNX:

| Grafo | Entrada → salida |
|---|---|
| `tok*_enc.onnx` | velas normalizadas → bits del tokenizador |
| `<modelo>_lm.onnx` | tokens nuevos + caché → logits del token s1 + caché ampliada |
| `<modelo>_s2.onnx` | estado + token s1 muestreado → logits del token s2 |
| `tok*_dec.onnx` | bits → velas normalizadas |

`src/engine/kronos.ts` ejecuta el bucle: el contexto se procesa una sola vez y se comparte entre
todas las trayectorias, y cada paso procesa un único token. En WebGPU la caché se queda en la GPU
entre pasos. El muestreo (temperatura, top-k, top-p) se hace en JavaScript con una semilla
reproducible. Si contexto + horizonte supera la ventana del modelo, la caché se reconstruye cada
64 pasos con los tokens más recientes; el original desliza la ventana un token cada paso, así que
a partir de ese punto las muestras no son idénticas a las suyas.

Chronos-Bolt es un único grafo (`bolt_*.onnx`) que devuelve nueve cuantiles del cierre para 64
pasos; los horizontes más largos se encadenan realimentando la mediana.

## Desarrollo

```bash
npm install
KRONOS_MODELS=/ruta/a/modelos npm run dev      # sirve /models/ desde esa carpeta
KRONOS_MODELS=/ruta/a/modelos npm test         # compara el motor con PyTorch (decodificación voraz)
npm run build
./deploy.sh [/ruta/a/modelos]                  # copia dist/ (y los modelos) a la carpeta que sirve la web
```

Los modelos no están en el repositorio. Para regenerarlos (Python 3.12, ~2 GB de disco):

```bash
git clone --depth 1 https://github.com/shiyu-coder/Kronos.git upstream
python3 -m venv venv && ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install numpy pandas einops huggingface_hub safetensors onnx onnxruntime tqdm chronos-forecasting
./venv/bin/python tools/export_kronos.py --upstream upstream --out models   # mini, small, base
./venv/bin/python tools/verify_kronos.py --upstream upstream --out models   # diferencia máx. ~4e-5
./venv/bin/python tools/export_chronos.py --out models                      # tiny, small
```

`models/manifest.json` describe cada archivo con su tamaño y SHA-256; la web lo usa para la
descarga, la caché (`?v=<sha>`) y los límites de cada modelo.

## Límites conocidos

- La ruta WebGPU se ha comprobado en Chromium con un adaptador por software (resultados iguales a
  los de CPU), no en una GPU real. En modo automático, si WebGPU falla se reintenta en CPU.
- Los modelos van en precisión de 32 bits; no hay variantes cuantizadas.
- Ningún modelo predice mercados de forma fiable: el backtest está para comprobarlo.

## Alojarlo

Basta un servidor de archivos estáticos que sirva `dist/` y, en `/models/`, los archivos exportados
con `manifest.json`. Opcionalmente, los pesos pueden descargarse de un CDN: define `VITE_MODELS_URL`
al compilar (ver `.env.production`); `manifest.json` se sigue leyendo de `/models/` y esa carpeta hace de
respaldo si el CDN falla. `tools/upload_r2.py` sube los modelos a un bucket de Cloudflare R2. Para que WebAssembly use varios hilos, todas las respuestas deben llevar
`Cross-Origin-Opener-Policy: same-origin` y `Cross-Origin-Embedder-Policy: require-corp`.
`deploy/nginx.conf` es la configuración usada en producción.

## Licencia

Código bajo licencia [MIT](LICENSE). Kronos (MIT), Chronos-Bolt (Apache-2.0), ONNX Runtime Web
(MIT), Inter (SIL OFL 1.1) y los iconos de Lucide (ISC) mantienen sus licencias; detalle en
[/licencias/](https://kronos.aminox.dev/licencias/).
