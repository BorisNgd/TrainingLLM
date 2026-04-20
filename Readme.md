# 🎫 Démonstration d'Entraînement de Modèle de Ticketing

Cette démonstration présente comment entraîner un modèle de langage réduit sur des données de système de ticketing, le tout dans un environnement Docker pour une présentation facilement reproductible.

## 📋 Vue d'ensemble

Cette preuve de concept démontre :
- ✅ Génération de données de ticketing simulées
- ✅ Entraînement d'un modèle de langage léger (DistilGPT-2)
- ✅ Interface web interactive avec Streamlit
- ✅ Déploiement containerisé avec Docker
- ✅ Test et évaluation du modèle entraîné

## 🏗️ Architecture

```
📦 Projet
├── 🐳 Dockerfile                 # Configuration du conteneur
├── 📝 docker-compose.yml         # Orchestration des services
├── 📋 requirements.txt           # Dépendances Python
├── 🚀 setup.sh                   # Script de configuration
├── 🎯 train_model.py             # Script d'entraînement
├── 🌐 app.py                     # Interface Streamlit
├── 📁 data/                      # Données générées
├── 🤖 models/                    # Modèles entraînés
├── 📊 logs/                      # Logs d'entraînement
└── 💾 cache/                     # Cache des modèles
```

## 🚀 Démarrage rapide

### Prérequis
- Docker (version 20+)
- Docker Compose (version 2+)
- 4GB RAM minimum
- 2GB d'espace disque libre

### Installation et lancement

1. **Cloner ou télécharger le projet**
```bash
# Si vous avez git
git clone <repo-url>
cd ticketing-ml-demo

# Ou extraire le ZIP téléchargé
```

2. **Lancer le script de configuration**
```bash
# Rendre le script exécutable
chmod +x setup.sh

# Lancer la configuration
./setup.sh
```

3. **Accéder à l'application**
- Ouvrir votre navigateur
- Aller à `http://localhost:8501`
- L'interface Streamlit se charge automatiquement

## 🎯 Guide d'utilisation

### Étape 1: Génération des données 📊
1. Dans l'interface, sélectionnez "📊 Génération des données"
2. Configurez le nombre de tickets (recommandé: 500-1000)
3. Cliquez sur "🎲 Générer les données"
4. Visualisez les statistiques générées

### Étape 2: Entraînement du modèle 🤖
1. Allez dans "🤖 Entraînement du modèle"
2. Configurez les paramètres :
   - Modèle de base: `distilgpt2` (recommandé)
   - Époques: 3 (pour démo rapide)
   - Batch size: 4
   - Learning rate: 5e-5
3. Cliquez sur "🚀 Lancer l'entraînement"
4. Attendez 5-15 minutes selon votre machine

### Étape 3: Test du modèle 🧪
1. Accédez à "🧪 Test et évaluation"
2. Testez avec vos propres données ou utilisez les tests prédéfinis
3. Observez les solutions générées par le modèle

### Étape 4: Visualisation des résultats 📈
1. Consultez "📈 Visualisation des résultats"
2. Analysez les métriques d'entraînement
3. Examinez les recommandations d'amélioration

## 🔧 Configuration avancée

### Personnalisation des données
Modifiez `train_model.py` dans la classe `TicketingDataGenerator` :
```python
# Ajoutez vos propres catégories
self.categories = [
    "Votre catégorie 1",
    "Votre catégorie 2",
    # ...
]

# Personnalisez les solutions
self.solutions = {
    "Votre catégorie 1": "Votre solution type...",
    # ...
}
```

### Paramètres d'entraînement
Dans `train_model.py`, classe `TicketingModelTrainer` :
```python
training_args = TrainingArguments(
    num_train_epochs=5,        # Plus d'époques
    per_device_train_batch_size=8,  # Batch plus grand
    learning_rate=1e-4,        # Taux d'apprentissage
    # ...
)
```

### Modèles alternatifs
Testez d'autres modèles dans l'interface :
- `gpt2` (plus grand, plus lent)
- `distilbert-base-uncased` (pour classification)

## 📊 Métriques et évaluation

### Métriques suivies
- **Train Loss** : Perte sur les données d'entraînement
- **Eval Loss** : Perte sur les données de validation
- **Improvement** : Pourcentage d'amélioration

### Interprétation des résultats
- **Perte < 2.0** : Bon apprentissage
- **Perte > 3.0** : Besoin de plus de données ou d'époques
- **Écart Train/Eval faible** : Pas de surapprentissage

## 🐳 Commandes Docker utiles

```bash
# Voir les logs en temps réel
docker-compose logs -f

# Redémarrer un service
docker-compose restart ticketing-ml-demo

# Arrêter tous les services
docker-compose down

# Reconstruire l'image
docker-compose build --no-cache

# Accéder au conteneur
docker-compose exec ticketing-ml-demo bash

# Nettoyer les volumes
docker-compose down -v
```

## 🔍 Dépannage

### Problèmes courants

**Erreur "Port 8501 already in use"**
```bash
# Arrêter le service existant
docker-compose down
# Ou changer le port dans docker-compose.yml
```

**Mémoire insuffisante**
```bash
# Réduire la taille du batch dans l'interface
# Ou allouer plus de RAM à Docker
```

**Modèle ne se charge pas**
```bash
# Vérifier l'espace disque
df -h
# Nettoyer le cache
docker system prune -a
```

### Logs et diagnostic
```bash
# Logs de l'application
docker-compose logs ticketing-ml-demo

# Utilisation des ressources
docker stats

# Espace disque des volumes
docker system df
```

## 📈 Optimisations pour production

### Performance
- Utiliser un GPU (modifier le Dockerfile pour CUDA)
- Augmenter la RAM allouée à Docker
- Utiliser des modèles plus grands (GPT-2 medium/large)

### Données réelles
- Remplacer les données simulées par vos vraies données
- Nettoyer et préprocesser les données de tickets
- Équilibrer les catégories de tickets

### Monitoring
```bash
# Activer le monitoring (optionnel)
docker-compose --profile monitoring up -d
# Accéder à Prometheus sur localhost:9090
```

## 🤝 Présentation en réunion

### Points clés à présenter
1. **Génération automatique de données** - Simulation réaliste de tickets
2. **Entraînement adaptatif** - Le modèle apprend de vos données spécifiques
3. **Interface intuitive** - Facile à utiliser et démontrer
4. **Déploiement simple** - Un seul command pour tout lancer
5. **Métriques visuelles** - Graphiques pour évaluer la performance

### Scénario de démonstration (15 minutes)
1. **2 min** : Présentation de l'architecture
2. **3 min** : Génération des données et visualisation
3. **5 min** : Lancement de l'entraînement (pendant la présentation)
4. **3 min** : Test du modèle avec exemples concrets
5. **2 min** : Métriques et recommandations

### Questions fréquentes
- **Temps d'entraînement ?** 5-15 minutes selon les données
- **Données nécessaires ?** Minimum 500 tickets, idéal 2000+
- **Précision attendue ?** 70-85% sur données similaires
- **Coût de déploiement ?** Minimal, fonctionne sur serveur standard

## 📞 Support

Pour toute question ou problème :
1. Vérifiez la section dépannage ci-dessus
2. Consultez les logs Docker
3. Examinez les fichiers générés dans `/data` et `/logs`

## 📝 Notes techniques

- **Modèle utilisé** : DistilGPT-2 (40M paramètres)
- **Framework** : Transformers de Hugging Face
- **Interface** : Streamlit
- **Containerisation** : Docker + Docker Compose
- **Données** : Format CSV avec prompts structurés

---

🎯 **Objectif** : Démontrer la faisabilité de l'entraînement de modèles sur des données de ticketing spécifiques à votre organisation.

✨ **Résultat** : Modèle capable de suggérer des solutions basées sur l'historique de résolution de tickets.