import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import sys
from train_model import TicketingDataGenerator, TicketingModelTrainer

# Configuration de la page
st.set_page_config(
    page_title="Démonstration - Entraînement Modèle de Ticketing",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎫 Démonstration d\'Entraînement de Modèle de Ticketing</h1>', unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    st.sidebar.title("🔧 Configuration")
    
    # Choix de l'étape
    step = st.sidebar.selectbox(
        "Choisissez une étape:",
        ["📊 Génération des données", "🤖 Entraînement du modèle", "🧪 Test et évaluation", "📈 Visualisation des résultats"]
    )
    
    if step == "📊 Génération des données":
        show_data_generation()
    elif step == "🤖 Entraînement du modèle":
        show_model_training()
    elif step == "🧪 Test et évaluation":
        show_model_testing()
    elif step == "📈 Visualisation des résultats":
        show_results_visualization()

def show_data_generation():
    st.header("📊 Génération des Données de Ticketing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration des données")
        
        num_tickets = st.slider(
            "Nombre de tickets à générer:",
            min_value=100,
            max_value=2000,
            value=500,
            step=100
        )
        
        if st.button("🎲 Générer les données", type="primary"):
            with st.spinner("Génération des données en cours..."):
                # Génération des données
                data_generator = TicketingDataGenerator()
                df = data_generator.generate_ticket_data(num_tickets=num_tickets)
                
                # Sauvegarde
                os.makedirs('data', exist_ok=True)
                df.to_csv('data/ticketing_data.csv', index=False)
                
                st.success(f"✅ {len(df)} tickets générés avec succès!")
                
                # Affichage des statistiques
                st.subheader("📋 Aperçu des données générées")
                
                # Métriques
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Tickets", len(df))
                with col_b:
                    st.metric("Catégories", df['category'].nunique())
                with col_c:
                    st.metric("Priorités", df['priority'].nunique())
                
                # Distribution des catégories
                st.subheader("📊 Distribution des catégories")
                fig, ax = plt.subplots(figsize=(10, 6))
                df['category'].value_counts().plot(kind='bar', ax=ax)
                plt.title("Distribution des catégories de tickets")
                plt.xlabel("Catégories")
                plt.ylabel("Nombre de tickets")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Échantillon de données
                st.subheader("🔍 Échantillon de données")
                st.dataframe(df.head(10))
    
    with col2:
        st.subheader("ℹ️ Information")
        st.info("""
        **Types de données générées:**
        
        📋 **Catégories:**
        - Problème technique
        - Demande d'information
        - Bug logiciel
        - Accès refusé
        - Performance lente
        - Erreur de connexion
        
        ⚡ **Priorités:**
        - Critique
        - Haute
        - Normale
        - Basse
        
        🎯 **Format de sortie:**
        - CSV avec prompts formatés
        - Prêt pour l'entraînement
        """)

def show_model_training():
    st.header("🤖 Entraînement du Modèle")
    
    # Vérification des données
    if not os.path.exists('data/ticketing_data.csv'):
        st.warning("⚠️ Aucune donnée trouvée. Veuillez d'abord générer les données.")
        return
    
    df = pd.read_csv('data/ticketing_data.csv')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration de l'entraînement")
        
        # Paramètres d'entraînement
        model_name = st.selectbox(
            "Modèle de base:",
            ["distilgpt2", "gpt2" , "microsoft/DialoGPT-medium" ,"benjamin/gpt2-wechsel-french" , "croissantllm/CroissantLLMChat-v0.1","microsoft/DialoGPT-medium"]
        )
        
        num_epochs = st.slider(
            "Nombre d'époques:",
            min_value=1,
            max_value=10,
            value=3
        )
        
        batch_size = st.selectbox(
            "Taille du batch:",
            [2, 4, 8, 16]
        )
        
        learning_rate = st.selectbox(
            "Taux d'apprentissage:",
            [1e-5, 5e-5, 1e-4, 5e-4]
        )
        
        if st.button("🚀 Lancer l'entraînement", type="primary"):
            with st.spinner("Entraînement en cours... Cela peut prendre plusieurs minutes."):
                try:
                    # Initialisation du trainer
                    trainer = TicketingModelTrainer(model_name=model_name)
                    
                    # Préparation du dataset
                    dataset = trainer.prepare_dataset(df)
                    
                    # Entraînement
                    training_history = trainer.train_model(dataset)
                    
                    st.success("✅ Entraînement terminé avec succès!")
                    
                    # Affichage des métriques
                    if training_history:
                        st.subheader("📊 Métriques d'entraînement")
                        
                        # Extraction des pertes
                        train_losses = [log['train_loss'] for log in training_history if 'train_loss' in log]
                        eval_losses = [log['eval_loss'] for log in training_history if 'eval_loss' in log]
                        
                        if train_losses:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(train_losses, label='Train Loss', marker='o')
                            if eval_losses:
                                ax.plot(eval_losses, label='Eval Loss', marker='s')
                            ax.set_xlabel('Étapes')
                            ax.set_ylabel('Perte')
                            ax.set_title('Évolution de la perte pendant l\'entraînement')
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                        
                        # Métriques finales
                        if train_losses:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Perte finale (Train)", f"{train_losses[-1]:.4f}")
                            with col_b:
                                if eval_losses:
                                    st.metric("Perte finale (Eval)", f"{eval_losses[-1]:.4f}")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
    
    with col2:
        st.subheader("📊 Données disponibles")
        st.metric("Tickets disponibles", len(df))
        
        st.subheader("⚙️ Paramètres recommandés")
        st.info("""
        **Pour une démonstration rapide:**
        - Modèle: distilgpt2
        - Époques: 2-3
        - Batch size: 4
        - Learning rate: 5e-5
        
        **Note:** L'entraînement peut prendre 5-15 minutes selon la configuration.
        """)

def show_model_testing():
    st.header("🧪 Test et Évaluation du Modèle")
    
    # Vérification du modèle entraîné
    if not os.path.exists('models/ticketing_model'):
        st.warning("⚠️ Aucun modèle entraîné trouvé. Veuillez d'abord entraîner un modèle.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Test du modèle")
        
        # Interface de test
        with st.form("test_form"):
            category = st.selectbox(
                "Catégorie du ticket:",
                ["Problème technique", "Demande d'information", "Bug logiciel",
                 "Accès refusé", "Performance lente", "Erreur de connexion"]
            )
            
            priority = st.selectbox(
                "Priorité:",
                ["Critique", "Haute", "Normale", "Basse"]
            )
            
            description = st.text_area(
                "Description du problème:",
                height=100,
                placeholder="Décrivez le problème rencontré..."
            )
            
            submitted = st.form_submit_button("🔮 Générer une solution")
        
        if submitted and description:
            with st.spinner("Génération de la solution..."):
                try:
                    # Chargement du modèle
                    trainer = TicketingModelTrainer()
                    
                    # Génération de la solution
                    solution = trainer.generate_solution(category, priority, description)
                    
                    st.success("✅ Solution générée!")
                    
                    # Affichage formaté
                    st.subheader("📋 Résultat")
                    
                    with st.container():
                        st.markdown("**Ticket d'entrée:**")
                        st.write(f"**Catégorie:** {category}")
                        st.write(f"**Priorité:** {priority}")
                        st.write(f"**Description:** {description}")
                        
                        st.markdown("---")
                        st.markdown("**Solution générée:**")
                        st.write(solution)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération: {str(e)}")
        
        # Tests prédéfinis
        st.subheader("🎲 Tests prédéfinis")
        
        test_cases = [
            {
                "name": "Bug critique",
                "category": "Bug logiciel",
                "priority": "Critique",
                "description": "Application qui plante au démarrage avec erreur 500"
            },
            {
                "name": "Performance lente",
                "category": "Performance lente",
                "priority": "Haute",
                "description": "Interface très lente, temps de chargement > 30 secondes"
            },
            {
                "name": "Accès refusé",
                "category": "Accès refusé",
                "priority": "Normale",
                "description": "Impossible d'accéder à la base de données clients"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            if st.button(f"🧪 Test: {test_case['name']}", key=f"test_{i}"):
                with st.spinner("Test en cours..."):
                    try:
                        trainer = TicketingModelTrainer()
                        solution = trainer.generate_solution(
                            test_case['category'],
                            test_case['priority'],
                            test_case['description']
                        )
                        
                        st.write(f"**{test_case['name']}**")
                        st.write(f"*{test_case['description']}*")
                        st.write(f"**Solution:** {solution}")
                        st.write("---")
                    
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
    
    with col2:
        st.subheader("ℹ️ Guide d'utilisation")
        st.info("""
        **Comment tester:**
        
        1. Sélectionnez une catégorie
        2. Choisissez une priorité
        3. Décrivez le problème
        4. Cliquez sur "Générer une solution"
        
        **Ou utilisez les tests prédéfinis** pour voir des exemples rapides.
        
        **Note:** La qualité des solutions dépend de:
        - La quantité de données d'entraînement
        - Le nombre d'époques
        - La similarité avec les données d'entraînement
        """)

def show_results_visualization():
    st.header("📈 Visualisation des Résultats")
    
    # Vérification des logs
    if not os.path.exists('logs/training_metrics.json'):
        st.warning("⚠️ Aucune métrique d'entraînement trouvée.")
        return
    
    # Chargement des métriques
    with open('logs/training_metrics.json', 'r') as f:
        training_history = json.load(f)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Métriques d'entraînement")
        
        # Extraction et visualisation des pertes
        train_losses = [log['train_loss'] for log in training_history if 'train_loss' in log]
        eval_losses = [log['eval_loss'] for log in training_history if 'eval_loss' in log]
        
        if train_losses:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(train_losses)), train_losses, label='Train Loss', marker='o')
            if eval_losses:
                eval_steps = [log['step'] for log in training_history if 'eval_loss' in log]
                ax.plot(eval_steps, eval_losses, label='Eval Loss', marker='s')
            
            ax.set_xlabel('Étapes')
            ax.set_ylabel('Perte')
            ax.set_title('Évolution de la perte')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # Métriques de performance
        if train_losses:
            st.subheader("🎯 Métriques de performance")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Perte initiale", f"{train_losses[0]:.4f}")
            with col_b:
                st.metric("Perte finale", f"{train_losses[-1]:.4f}")
            with col_c:
                improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
                st.metric("Amélioration", f"{improvement:.1f}%")
    
    with col2:
        st.subheader("📊 Analyse des données")
        
        # Vérification des données
        if os.path.exists('data/ticketing_data.csv'):
            df = pd.read_csv('data/ticketing_data.csv')
            
            # Distribution des catégories
            fig, ax = plt.subplots(figsize=(8, 6))
            df['category'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Distribution des catégories de tickets')
            ax.set_ylabel('')
            st.pyplot(fig)
            
            # Tableau de statistiques
            st.subheader("📋 Statistiques des données")
            stats_df = pd.DataFrame({
                'Catégorie': df['category'].value_counts().index,
                'Nombre': df['category'].value_counts().values,
                'Pourcentage': (df['category'].value_counts().values / len(df) * 100).round(1)
            })
            st.dataframe(stats_df)
    
    # Section résumé
    st.subheader("📝 Résumé de l'entraînement")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        if os.path.exists('data/ticketing_data.csv'):
            df = pd.read_csv('data/ticketing_data.csv')
            st.metric("Données d'entraînement", f"{len(df)} tickets")
    
    with summary_col2:
        if train_losses:
            st.metric("Époques complétées", len([log for log in training_history if 'epoch' in log]))
    
    with summary_col3:
        model_size = "~40MB" if os.path.exists('models/ticketing_model') else "N/A"
        st.metric("Taille du modèle", model_size)
    
    # Recommandations
    st.subheader("💡 Recommandations pour l'amélioration")
    
    recommendations = []
    
    if train_losses and len(train_losses) > 1:
        recent_improvement = train_losses[-2] - train_losses[-1]
        if recent_improvement < 0.01:
            recommendations.append("⚠️ La perte converge lentement. Considérez augmenter le taux d'apprentissage.")
        if train_losses[-1] > 2.0:
            recommendations.append("📊 La perte finale est élevée. Plus de données d'entraînement pourraient aider.")
    
    if os.path.exists('data/ticketing_data.csv'):
        df = pd.read_csv('data/ticketing_data.csv')
        if len(df) < 1000:
            recommendations.append("📈 Augmentez le nombre de tickets d'entraînement pour de meilleurs résultats.")
        
        category_balance = df['category'].value_counts()
        if category_balance.max() / category_balance.min() > 3:
            recommendations.append("⚖️ Les catégories sont déséquilibrées. Équilibrez les données d'entraînement.")
    
    if not recommendations:
        recommendations.append("✅ L'entraînement semble bien configuré!")
    
    for rec in recommendations:
        st.write(rec)

if __name__ == "__main__":
    main()