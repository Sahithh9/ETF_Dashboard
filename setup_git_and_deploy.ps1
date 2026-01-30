Write-Host "--- Streamlit Deployment Helper ---" -ForegroundColor Cyan

# Check if git is installed
$gitInstalled = $false
try {
    git --version | Out-Null
    $gitInstalled = $true
} catch {
    # Git not found
}

if (-not $gitInstalled) {
    Write-Host "Git is not detected." -ForegroundColor Yellow
    
    # Check for winget
    $wingetAvailable = $false
    try {
        winget --version | Out-Null
        $wingetAvailable = $true
    } catch {
        # Winget not found
    }

    if ($wingetAvailable) {
        Write-Host "Installing Git via Winget..." -ForegroundColor Cyan
        winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
        
        Write-Host "`nGit installation complete!" -ForegroundColor Green
        Write-Host "CRITICAL: You must RESTART VS Code (close and reopen) for the changes to take effect." -ForegroundColor Red
        Write-Host "After restarting, run this script again."
        pause
        exit
    } else {
        Write-Error "Git is not installed and Winget is not available. Please install manually: https://git-scm.com/downloads"
        pause
        exit
    }
}

Write-Host "1. Initializing Git..."
if (-not (Test-Path ".git")) {
    git init
} else {
    Write-Host "   Git already initialized."
}

Write-Host "2. Adding files..."
git add .

Write-Host "3. Committing files..."
git commit -m "Prepare for Streamlit Community Cloud deployment"

Write-Host "`n-----------------------------------------------------" -ForegroundColor Green
Write-Host "SUCCESS: Local repository prepared!" -ForegroundColor Green
Write-Host "-----------------------------------------------------" -ForegroundColor Green
Write-Host "NEXT STEPS (Manual Action Required):" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new and create a PUBLIC repository."
Write-Host "   (Name it something like 'micro-ind-dashboard')"
Write-Host "2. Copy the 'HTTPS' link of your new repository."
Write-Host "3. Paste the commands below into this terminal:"
Write-Host "`n   git remote add origin <PASTE_YOUR_REPO_URL_HERE>" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host "`nAfter pushing, go to https://share.streamlit.io/ to deploy."
Write-Host "-----------------------------------------------------`n"
pause
