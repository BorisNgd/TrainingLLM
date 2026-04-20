# Dockerfile pour l'entraînement d'un modèle de langage
#FROM python:3.9-slim
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt


# Copie du code source
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p data models logs

# Exposition du port pour Streamlit
EXPOSE 8501

# Variable d'environnement
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/cache

# Commande par défaut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]