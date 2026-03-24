# Web client (Streamlit)

Simple UI for the FastAPI backend in `app/main.py`.

## Features

- Upload image (`jpg`, `jpeg`, `png`, `webp`)
- Select fixed ROI `224x224`
- Preview selected area
- Send ROI to `POST /predecir`
- Show class, confidence and top-3

## Run

1. Start API:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Install web dependencies:

```bash
pip install -r clients/web/requirements.txt
```

3. Start Streamlit app:

```bash
streamlit run clients/web/app.py
```

The app calls `/predecir` on the configured API base URL.
In Docker Compose, `API_BASE_URL=http://api:8000`.

## Run with Docker Compose + Traefik (HTTPS)

From the project root:

```bash
docker compose up --build
```

Then open:

- `https://localhost` for Streamlit
- `https://localhost/api/docs` for FastAPI docs

Traefik terminates TLS at `:443` and forwards traffic to `web` and `api`.
