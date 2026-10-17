# Launch WBC Classification Streamlit App
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Launching White Blood Cell Classification Streamlit App..." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m streamlit run app/app.py
