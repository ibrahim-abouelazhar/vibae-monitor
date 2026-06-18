import streamlit as st
import json
from kafka import KafkaProducer
from datetime import datetime

st.set_page_config(
    page_title="VibAE-Monitor Live Control Panel",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b26; }
    .command-card { 
        background: #161b26; 
        border-radius: 8px; 
        padding: 12px 18px; 
        margin-bottom: 8px; 
        border-left: 4px solid #00FF66;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Title ───────────────────────────────────────────────────────────────────
st.title("🕹️ VibAE-Monitor · Live Control Panel")
st.caption("Injecteur de pannes en direct pour le flux Apache Kafka (Pompe, Ventilateur, Compresseur)")

# ── Kafka Command Producer (cached, singleton) ───────────────────────────────
@st.cache_resource
def get_command_producer():
    try:
        return KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except Exception as e:
        return None

cmd_producer = get_command_producer()

# ── Session State for Command History ────────────────────────────────────────
if "command_history" not in st.session_state:
    st.session_state.command_history = []

def log_command(machine, command):
    st.session_state.command_history.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine": machine.upper(),
        "command": command
    })

# ── Sidebar Anomaly Injection ───────────────────────────────────────────────
st.sidebar.header("🕹️ Contrôle du Flux IoT")
st.sidebar.write("Sélectionnez une machine et l'état à injecter :")

selected_machine = st.sidebar.selectbox("Machine cible :", ["Pompe", "Ventilateur", "Compresseur"])
state_options = ["Sain", "Inner Race Fault", "Ball Defect"]
selected_state = st.sidebar.radio("État du signal émis :", state_options)

if st.sidebar.button("▶ Appliquer l'état", use_container_width=True):
    if cmd_producer:
        try:
            cmd_producer.send("sensor-command", {"machine": selected_machine.lower(), "command": selected_state})
            cmd_producer.flush()
            log_command(selected_machine, selected_state)
            st.sidebar.success(f"✅ Commande émise : **{selected_state}**")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur d'émission : {e}")
    else:
        st.sidebar.error("❌ Kafka Producer indisponible")

if st.sidebar.button("🔧 Réparer", use_container_width=True):
    if cmd_producer:
        try:
            cmd_producer.send("sensor-command", {"machine": selected_machine.lower(), "command": "Sain"})
            cmd_producer.flush()
            log_command(selected_machine, "Réparation")
            st.sidebar.success(f"✅ Commande émise : **Réparation** (Sain)")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de réparation : {e}")
    else:
        st.sidebar.error("❌ Kafka Producer indisponible")

st.sidebar.divider()
st.sidebar.subheader("🤖 Réinitialisation Individuelle")
for machine in ["pompe", "ventilateur", "compresseur"]:
    if st.sidebar.button(f"Reset Auto ({machine.capitalize()})", key=f"reset_{machine}", use_container_width=True):
        if cmd_producer:
            try:
                cmd_producer.send("sensor-command", {"machine": machine, "command": "Reset Auto"})
                cmd_producer.flush()
                log_command(machine, "Reset Auto (Auto-Reroll)")
                st.sidebar.success(f"Re-rolled {machine.capitalize()}!")
            except Exception as e:
                st.sidebar.error(f"❌ Erreur reset : {e}")
        else:
            st.sidebar.error("❌ Kafka Producer indisponible")

st.sidebar.divider()
st.sidebar.info(
    "**Status du Pipeline :**\n"
    "- 🟢 Envoie sur le topic `sensor-command`\n"
    "- Visualisez le résultat en temps réel sur la page principale dashboard.html (port 8080)"
)

# ── Main Body Command Log ───────────────────────────────────────────────────
st.subheader("📋 Historique des Commandes Émises")
if st.session_state.command_history:
    for cmd in st.session_state.command_history:
        color = "#00FF66" if "Sain" in cmd["command"] or "Reset" in cmd["command"] or "Réparation" in cmd["command"] else "#FF3366"
        st.markdown(
            f"""
            <div class="command-card" style="border-left: 4px solid {color};">
                <strong>{cmd['timestamp']}</strong> — Machine: <code>{cmd['machine']}</code> &nbsp;|&nbsp; 
                Action: <span style="color:{color}; font-weight:bold;">{cmd['command']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Aucune commande émise pour le moment. Utilisez le panneau latéral pour injecter des signaux.")
