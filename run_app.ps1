# Run the Streamlit chatbot (Windows). From project root:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # if scripts are blocked
#   .\run_app.ps1

Set-Location $PSScriptRoot
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Create venv first: python -m venv .venv"
    Write-Host "Then: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
