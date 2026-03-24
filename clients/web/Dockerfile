FROM python:3.10-slim

WORKDIR /app

COPY clients/web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY clients/web/app.py /app/app.py

EXPOSE 8501

CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
