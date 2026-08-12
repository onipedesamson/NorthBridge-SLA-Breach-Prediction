FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.120.2 \
    uvicorn==0.30.6 \
    pydantic==2.11.10 \
    pandas==2.2.3 \
    joblib==1.4.2 \
    scikit-learn==1.7.2 \
    numpy==1.26.4

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]