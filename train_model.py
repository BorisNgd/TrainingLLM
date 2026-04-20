import os
import json
import torch
import pandas as pd
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

class TicketingDataGenerator:
    """Générateur de données de ticketing simulées"""
    
    def __init__(self):
        self.categories = [
            "Problème technique", "Demande d'information", "Bug logiciel",
            "Accès refusé", "Performance lente", "Erreur de connexion",
            "Mise à jour requise", "Formation utilisateur", "Récupération de mot de passe"
        ]
        
        self.priorities = ["Critique", "Haute", "Normale", "Basse"]
        
        self.solutions = {
            "Problème technique": "Vérification des logs système et redémarrage du service concerné.",
            "Demande d'information": "Fourniture de la documentation appropriée et explication détaillée.",
            "Bug logiciel": "Identification du bug, correction dans le code et déploiement du patch.",
            "Accès refusé": "Vérification des permissions utilisateur et mise à jour des droits d'accès.",
            "Performance lente": "Optimisation des requêtes et vérification de la charge serveur.",
            "Erreur de connexion": "Test de connectivité réseau et reconfiguration des paramètres.",
            "Mise à jour requise": "Planification et exécution de la mise à jour système.",
            "Formation utilisateur": "Organisation de session de formation et création de guides."
        }
    
    def generate_ticket_data(self, num_tickets=1000):
        """Génère des données de tickets simulées"""
        import random
        
        tickets = []
        for i in range(num_tickets):
            category = random.choice(self.categories)
            priority = random.choice(self.priorities)
            
            # Génération de la description du problème
            # Correction: éviter les backslashes dans les f-strings
            app_choices = ["l'application", "le système", "l'interface"]
            action_choices = ["configurer", "utiliser", "comprendre"]
            system_choices = ["le système", "l'application", "la fonctionnalité"]
            process_choices = ["la connexion", "l'utilisation", "la sauvegarde"]
            access_choices = ["la base de données", "l'application", "le serveur"]
            response_choices = ["la recherche", "l'affichage", "la connexion"]
            server_choices = ["web", "database", "email"]
            
            descriptions = {
                "Problème technique": f"L'utilisateur {i+1} rencontre un dysfonctionnement avec {random.choice(app_choices)}.",
                "Demande d'information": f"Demande d'aide pour {random.choice(action_choices)} {random.choice(system_choices)}.",
                "Bug logiciel": f"Erreur détectée lors de {random.choice(process_choices)} - Code erreur: {random.randint(100, 999)}.",
                "Accès refusé": f"Impossible d'accéder à {random.choice(access_choices)} - Permissions insuffisantes.",
                "Performance lente": f"Temps de réponse très élevé lors de {random.choice(response_choices)} - {random.randint(5, 30)} secondes.",
                "Erreur de connexion": f"Échec de connexion au serveur {random.choice(server_choices)} - Timeout après {random.randint(30, 120)} secondes."
            }
            
            description = descriptions.get(category, f"Problème de catégorie {category} pour l'utilisateur {i+1}.")
            solution = self.solutions.get(category, "Solution en cours d'élaboration.")
            
            # Format de prompt pour l'entraînement
            prompt = f"""Catégorie: {category}
Priorité: {priority}
Description: {description}
Solution: {solution}
Statut: Résolu"""
            
            tickets.append({
                'id': f'TICKET-{i+1:04d}',
                'category': category,
                'priority': priority,
                'description': description,
                'solution': solution,
                'prompt': prompt
            })
        
        return pd.DataFrame(tickets)

class TicketingModelTrainer:
    """Classe pour l'entraînement du modèle sur les données de ticketing"""
    
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ajout du token de padding si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, df):
        """Prépare le dataset pour l'entraînement"""
        # Tokenisation des prompts
        def tokenize_function(examples):
            return self.tokenizer(
                examples["prompt"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_pandas(df[['prompt']])
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_model(self, dataset, output_dir="./models/ticketing_model"):
        """Entraîne le modèle sur le dataset"""
        
        # Configuration de l'entraînement
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir='./logs',
            report_to=None,  # Désactive wandb pour simplifier
            save_total_limit=2,
            load_best_model_at_end=True,
        )
        
        # División du dataset
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        # Collator pour les données
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Création du trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Entraînement
        print("Début de l'entraînement...")
        trainer.train()
        
        # Sauvegarde du modèle
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Modèle sauvegardé dans {output_dir}")
        
        return trainer.state.log_history
    
    def generate_solution(self, category, priority, description, max_length=150):
        """Génère une solution pour un nouveau ticket"""
        prompt = f"""Catégorie: {category}
Priorité: {priority}
Description: {description}
Solution:"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraction de la solution générée
        if "Solution:" in generated_text:
            solution = generated_text.split("Solution:")[-1].strip()
            return solution
        
        return "Solution non générée"

def main():
    """Fonction principale d'entraînement"""
    print("🎫 Démonstration d'entraînement - Modèle de Ticketing")
    print("=" * 60)
    
    # Créer les dossiers nécessaires s'ils n'existent pas
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Génération des données
    print("\n📊 Génération des données de ticketing...")
    data_generator = TicketingDataGenerator()
    df = data_generator.generate_ticket_data(num_tickets=500)  # Réduit pour la démo
    
    print(f"✅ {len(df)} tickets générés")
    print("\n📋 Aperçu des données:")
    print(df[['category', 'priority']].value_counts().head(10))
    
    # Sauvegarde des données
    df.to_csv('data/ticketing_data.csv', index=False)
    
    # 2. Entraînement du modèle
    print("\n🤖 Initialisation du modèle...")
    trainer = TicketingModelTrainer()
    
    print("\n🔄 Préparation du dataset...")
    dataset = trainer.prepare_dataset(df)
    
    print("\n🏋️ Début de l'entraînement...")
    training_history = trainer.train_model(dataset)
    
    # 3. Test du modèle
    print("\n🧪 Test du modèle entraîné...")
    test_cases = [
        {
            "category": "Bug logiciel",
            "priority": "Critique",
            "description": "Application qui plante au démarrage avec erreur 404"
        },
        {
            "category": "Performance lente",
            "priority": "Haute",
            "description": "Interface très lente, temps de chargement > 20 secondes"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Catégorie: {test_case['category']}")
        print(f"Priorité: {test_case['priority']}")
        print(f"Description: {test_case['description']}")
        
        solution = trainer.generate_solution(
            test_case['category'],
            test_case['priority'],
            test_case['description']
        )
        print(f"Solution générée: {solution}")
    
    # 4. Sauvegarde des métriques
    if training_history:
        with open('logs/training_metrics.json', 'w') as f:
            json.dump(training_history, f, indent=2)
    
    print("\n✅ Entraînement terminé avec succès!")
    print("📁 Fichiers générés:")
    print("   - data/ticketing_data.csv")
    print("   - models/ticketing_model/")
    print("   - logs/training_metrics.json")

if __name__ == "__main__":
    main()