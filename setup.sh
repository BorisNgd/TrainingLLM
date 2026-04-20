#!/bin/bash

echo "🎫 Configuration de la démonstration - Modèle de Ticketing"
echo "========================================================="

# Création des répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p data models logs cache monitoring

# Vérification de Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez installer Docker d'abord."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Veuillez installer Docker Compose d'abord."
    exit 1
fi

echo "✅ Docker et Docker Compose sont installés"

# Construction de l'image Docker
echo "🔨 Construction de l'image Docker..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Image Docker construite avec succès"
else
    echo "❌ Erreur lors de la construction de l'image Docker"
    exit 1
fi

# Démarrage des services
echo "🚀 Démarrage des services..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services démarrés avec succès"
    echo ""
    echo "🌐 Application disponible sur:"
    echo "   http://localhost:8501"
    echo ""
    echo "📋 Commandes utiles:"
    echo "   - Voir les logs: docker-compose logs -f"
    echo "   - Arrêter: docker-compose down"
    echo "   - Redémarrer: docker-compose restart"
    echo ""
    echo "🎯 Pour accéder à l'application, ouvrez votre navigateur et allez sur http://localhost:8501"
else
    echo "❌ Erreur lors du démarrage des services"
    exit 1
fi