# VibAE-Monitor MLOps Distributed Pipeline Launcher
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Starting VibAE-Monitor MLOps Kafka Pipeline  " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
docker ps > $null 2>&1
if ($LastExitCode -ne 0) {
    Write-Host "Docker is not running or not accessible." -ForegroundColor Red
    Write-Host "Please start Docker Desktop manually and wait for the engine to initialize." -ForegroundColor Red
    Read-Host "Press Enter once Docker Desktop is running to continue..."
}

# 2. Start Zookeeper & Kafka
Write-Host "Starting Kafka and Zookeeper via Docker Compose..." -ForegroundColor Yellow
docker compose -f dist_mlops/docker-compose.yml up -d
if ($LastExitCode -ne 0) {
    Write-Host "Failed to start Docker containers. Exiting." -ForegroundColor Red
    Exit 1
}

Write-Host "Waiting 10 seconds for Kafka broker to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# 3. Start Producer in a new terminal
Write-Host "Starting Edge Sensor Simulator (producer.py) in a new window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- IoT Edge Sensor Simulator ---' -ForegroundColor Green; .venv\Scripts\python dist_mlops\producer.py"

# 4. Start Consumer in a new terminal
Write-Host "Starting PyTorch Inference Consumer (consumer.py) in a new window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- PyTorch Inference Consumer ---' -ForegroundColor Green; .venv\Scripts\python dist_mlops\consumer.py"

Write-Host "Waiting 5 seconds for model initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 5. Start Streamlit Dashboard in a new terminal
Write-Host "Starting Streamlit Dashboard (app.py) in a new window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- Streamlit Live Dashboard ---' -ForegroundColor Green; .venv\Scripts\streamlit run dist_mlops\app.py"

Write-Host "===============================================" -ForegroundColor Green
Write-Host "Pipeline components started. Check the new windows." -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
