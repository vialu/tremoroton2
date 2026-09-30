# TSI App — cómo generar el ejecutable para Windows y Mac

Esta carpeta contiene todo lo necesario para que **GitHub** compile automáticamente
dos versiones de la app — un `.exe` para Windows y un `.app` para Mac — sin que tengas
que instalar nada en tu computador. Tú solo subes estos archivos una vez; GitHub hace
el resto en la nube cada vez que lo actualicemos.

## Qué hay en esta carpeta

- `tsi_app.py` — el script de la aplicación (la misma app que ya conoces).
- `requirements.txt` — la lista de librerías que necesita.
- `.github/workflows/build.yml` — la "receta" que le dice a GitHub cómo compilarla.

## Paso 1 — Crear un repositorio en GitHub (una sola vez)

1. Entra a [github.com](https://github.com) y crea una cuenta gratuita si no tienes una.
2. Arriba a la derecha, clic en **"+"** → **"New repository"**.
3. Ponle un nombre, por ejemplo `tsi-app`. Puedes dejarlo como **Private** (privado) —
   solo tú (y quien invites) lo verá.
4. Clic en **"Create repository"**.

## Paso 2 — Subir los archivos

1. En la página del repositorio recién creado, clic en **"uploading an existing file"**
   (o **"Add file" → "Upload files"**).
2. Arrastra **toda esta carpeta** (`tsi_app.py`, `requirements.txt`, y la carpeta `.github`
   completa con su contenido) a la ventana del navegador.
   - Importante: la carpeta `.github/workflows/build.yml` debe mantener exactamente esa
     ruta/estructura — si tu navegador no te deja arrastrar carpetas, usa el botón
     "choose your files" y selecciona los archivos manteniendo esa misma estructura.
3. Abajo, clic en **"Commit changes"**.

## Paso 3 — Ver el build automático

1. En la parte superior del repositorio, clic en la pestaña **"Actions"**.
2. Deberías ver un proceso llamado **"Build TSI App (Windows + Mac)"** corriendo
   (círculo amarillo/naranja = en progreso). Tarda entre 5 y 10 minutos.
3. Si por algún motivo no arrancó solo, hay un botón **"Run workflow"** en esa misma
   pestaña para lanzarlo manualmente.

## Paso 4 — Descargar los archivos listos

1. Cuando el círculo se ponga **verde** (✓), haz clic en ese build.
2. Abajo, en la sección **"Artifacts"**, vas a ver dos archivos para descargar:
   - **TSI_App-Windows** → contiene `TSI_App.exe`
   - **TSI_App-Mac** → contiene `TSI_App-Mac.zip` (descomprímelo para obtener `TSI_App.app`)
3. Esos son los archivos que le compartes a tus colegas — se abren con doble clic,
   sin instalar Python ni nada más.

## Cosas a tener en cuenta

- **Primera vez que alguien lo abre**: como el archivo no está "firmado" digitalmente
  (eso tiene un costo y trámite aparte), Windows puede mostrar *"Windows protegió tu PC"*
  y Mac puede decir que es de *"un desarrollador no identificado"*. No es un virus — es
  la advertencia estándar para cualquier programa no firmado. Se destraba así:
  - **Windows**: clic en "Más información" → "Ejecutar de todas formas".
  - **Mac**: clic derecho sobre el archivo → "Abrir" → confirmar "Abrir" en el diálogo
    (en vez de hacer doble clic directamente la primera vez).
- **Primer arranque es más lento** (unos segundos): el `.exe`/`.app` se "descomprime" a
  sí mismo en memoria cada vez que abre — es normal para este tipo de empaquetado.
- **Actualizaciones futuras**: cuando quieras una nueva versión, solo reemplaza
  `tsi_app.py` en el repositorio (mismo botón "Upload files", sobrescribe el archivo) y
  el build se vuelve a correr solo.
- **Si el build falla** (círculo rojo ✗ en Actions): clic en el build fallido, copia el
  texto del error y compártemelo — casi siempre es un ajuste de una línea en
  `build.yml` (por ejemplo, una librería que necesita una bandera extra para empaquetarse
  bien) y lo corrijo sin tener que tocar tu computador.
