Write-Host "--- Streamlit Update Helper ---" -ForegroundColor Cyan

Write-Host "1. Staging changes..."
git add .

Write-Host "2. Committing fixes..."
git commit -m "Fix: Handle empty data crash and upgrade yfinance"

Write-Host "3. Pushing to GitHub..."
git push

Write-Host "`n-----------------------------------------------------" -ForegroundColor Green
Write-Host "SUCCESS: Updates pushed!" -ForegroundColor Green
Write-Host "-----------------------------------------------------" -ForegroundColor Green
Write-Host "Streamlit Cloud should automatically detect these changes"
Write-Host "and reboot your app in 1-2 minutes."
Write-Host "-----------------------------------------------------`n"
pause
