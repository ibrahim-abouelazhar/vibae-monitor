# Walkthrough - separation of controls and layout optimization

We have successfully separated the control responsibilities and resolved layout overflow issues in the dashboard:

## Changes Implemented

### 1. Frontend Layout & Card Updates (`dashboard.html`)
- Re-styled the cards layout container from `.factory-floor` (4 columns) to `.machine-cards-row` (3 columns grid layout).
- Updated cards to prevent layout overflow by specifying `min-width: 0` and `overflow: hidden`.
- Removed the Kafka Live state selection dropdowns and "Appliquer" buttons from Pompe, Ventilateur, and Compresseur cards. The cards are now pure display for Kafka Live mode (only showing the active filename, MSE, diagnosis, status dot, and color border on anomaly).
- Retained CWRU Fichier mode manual controls unchanged as requested.

### 2. IoT Simulation Updates (`producer.py`)
- Configured the producer to pre-load all 10 `.mat` files from CWRU data directory at startup.
- Implemented `roll_random_file()` which picks a normal file with 95% probability (`Time_Normal_1_098.mat`) and a fault file with 5% probability (random selection from the 9 fault files).
- Each machine thread auto-picks its active file randomly using this distribution at producer start.
- Updated the Kafka command listener to handle:
  - Global manual override command options (from 8502 app): maps `Sain`, `Inner Race Fault`, `Ball Defect` states to representative files and applies them to all machines.
  - Individual machine commands: supports `"Reset Auto"` command, which re-rolls the machine file randomly using the 95%/5% distribution.

### 3. Streamlit Dashboard Updates (`app.py`)
- Updated the Streamlit alert listener to subscribe to all three machine-specific alert topics (`alert-pompe`, `alert-ventilateur`, `alert-compresseur`).
- Added individual re-roll controls under the sidebar section `🤖 Réinitialisation Individuelle`. For each machine, a "Reset Auto" button triggers a Kafka command to re-roll its active file.

---

## Visual Verification

Here is the updated dashboard layout on port 8080:

![Dashboard Fichier Mode Layout](/c:/Users/HP/.gemini/antigravity-ide/brain/47ae7c28-1675-423d-b937-1991d9a49877/dashboard_fichier_1781741684698.png)
*Figure 1: Machine cards in a clean 3-column grid row, with Kafka controls removed.*
