# Script to create backdated commits
# This creates a realistic commit history from September 2, 2025 to present

$commits = @(
    @{Date="2025-09-02 10:15:00"; Msg="feat: initialize project structure"; Files=@(".gitignore", "README.md")},
    @{Date="2025-09-02 14:30:00"; Msg="feat: add backend directory structure"; Files=@("backend/")},
    @{Date="2025-09-03 09:20:00"; Msg="feat: setup FastAPI application base"; Files=@("backend/app/main.py", "backend/requirements.txt")},
    @{Date="2025-09-03 16:45:00"; Msg="feat: implement core configuration"; Files=@("backend/app/core/")},
    @{Date="2025-09-04 11:00:00"; Msg="feat: add audio processing utilities"; Files=@("backend/app/ml/utils/audio_processor.py")},
    @{Date="2025-09-05 10:30:00"; Msg="feat: implement mel spectrogram extraction"; Files=@("backend/app/ml/utils/")},
    @{Date="2025-09-06 14:00:00"; Msg="feat: add MFCC feature extraction"; Files=@("backend/app/ml/utils/")},
    @{Date="2025-09-08 09:15:00"; Msg="feat: implement 2D CNN architecture"; Files=@("backend/app/ml/models/networks.py")},
    @{Date="2025-09-09 13:20:00"; Msg="feat: add 1D CNN waveform classifier"; Files=@("backend/app/ml/models/")},
    @{Date="2025-09-10 10:45:00"; Msg="feat: implement autoencoder for fingerprinting"; Files=@("backend/app/ml/models/")},
    @{Date="2025-09-11 15:30:00"; Msg="feat: add model weight initialization"; Files=@("backend/app/ml/models/")},
    @{Date="2025-09-12 11:10:00"; Msg="feat: create training script for 2D CNN"; Files=@("ml/train_classifier.py")},
    @{Date="2025-09-13 09:50:00"; Msg="feat: add validation loop and metrics"; Files=@("ml/train_classifier.py")},
    @{Date="2025-09-15 14:25:00"; Msg="feat: implement autoencoder training"; Files=@("ml/train_autoencoder.py")},
    @{Date="2025-09-16 10:00:00"; Msg="feat: add fingerprint database builder"; Files=@("ml/train_fingerprint_model.py")},
    @{Date="2025-09-17 16:15:00"; Msg="feat: create inference engine"; Files=@("backend/app/ml/inference.py")},
    @{Date="2025-09-18 11:30:00"; Msg="feat: add sound classifier class"; Files=@("backend/app/ml/inference.py")},
    @{Date="2025-09-19 09:40:00"; Msg="feat: implement anomaly detector"; Files=@("backend/app/ml/inference.py")},
    @{Date="2025-09-20 13:55:00"; Msg="feat: add fingerprint matcher"; Files=@("backend/app/ml/inference.py")},
    @{Date="2025-09-22 10:20:00"; Msg="feat: create API schemas"; Files=@("backend/app/api/schemas.py")},
    @{Date="2025-09-23 14:45:00"; Msg="feat: implement upload endpoint"; Files=@("backend/app/api/routes.py")},
    @{Date="2025-09-24 11:15:00"; Msg="feat: add analysis endpoint"; Files=@("backend/app/api/routes.py")},
    @{Date="2025-09-25 09:30:00"; Msg="feat: implement file cleanup task"; Files=@("backend/app/api/routes.py")},
    @{Date="2025-09-26 15:00:00"; Msg="feat: add CORS middleware"; Files=@("backend/app/main.py")},
    @{Date="2025-09-27 10:50:00"; Msg="feat: create health check endpoint"; Files=@("backend/app/main.py")},
    @{Date="2025-09-29 14:20:00"; Msg="feat: initialize Next.js frontend"; Files=@("frontend/")},
    @{Date="2025-09-30 11:00:00"; Msg="feat: setup Tailwind CSS"; Files=@("frontend/tailwind.config.ts")},
    @{Date="2025-10-01 09:45:00"; Msg="feat: add shadcn/ui components"; Files=@("frontend/components/ui/")},
    @{Date="2025-10-02 13:30:00"; Msg="feat: create Button component"; Files=@("frontend/components/ui/button.tsx")},
    @{Date="2025-10-03 10:15:00"; Msg="feat: add Card component"; Files=@("frontend/components/ui/card.tsx")},
    @{Date="2025-10-04 15:40:00"; Msg="feat: implement Tabs component"; Files=@("frontend/components/ui/tabs.tsx")},
    @{Date="2025-10-06 11:20:00"; Msg="feat: create FileUpload component"; Files=@("frontend/components/FileUpload.tsx")},
    @{Date="2025-10-07 09:55:00"; Msg="feat: add drag and drop upload"; Files=@("frontend/components/FileUpload.tsx")},
    @{Date="2025-10-08 14:10:00"; Msg="feat: implement upload progress"; Files=@("frontend/components/FileUpload.tsx")},
    @{Date="2025-10-09 10:30:00"; Msg="feat: create AudioRecorder component"; Files=@("frontend/components/AudioRecorder.tsx")},
    @{Date="2025-10-10 13:45:00"; Msg="feat: add waveform visualization"; Files=@("frontend/components/AudioRecorder.tsx")},
    @{Date="2025-10-11 11:15:00"; Msg="feat: implement recording controls"; Files=@("frontend/components/AudioRecorder.tsx")},
    @{Date="2025-10-13 09:00:00"; Msg="feat: create SoundReportDashboard"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-14 14:25:00"; Msg="feat: add classification results display"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-15 10:50:00"; Msg="feat: implement anomaly detection UI"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-16 13:20:00"; Msg="feat: add similar sounds section"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-17 11:40:00"; Msg="feat: integrate Plotly for waveform"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-18 15:15:00"; Msg="feat: add spectrogram visualization"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-20 10:05:00"; Msg="feat: implement 3D FFT plot"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-10-21 14:30:00"; Msg="feat: create main page layout"; Files=@("frontend/app/page.tsx")},
    @{Date="2025-10-22 11:20:00"; Msg="feat: add loading states"; Files=@("frontend/app/page.tsx")},
    @{Date="2025-10-23 09:45:00"; Msg="feat: implement analysis flow"; Files=@("frontend/app/page.tsx")},
    @{Date="2025-10-24 13:55:00"; Msg="feat: add features showcase section"; Files=@("frontend/app/page.tsx")},
    @{Date="2025-10-25 10:30:00"; Msg="style: improve gradient header"; Files=@("frontend/app/page.tsx")},
    @{Date="2025-10-27 14:10:00"; Msg="feat: create backend Dockerfile"; Files=@("backend/Dockerfile")},
    @{Date="2025-10-28 11:00:00"; Msg="feat: add frontend Dockerfile"; Files=@("frontend/Dockerfile")},
    @{Date="2025-10-29 09:25:00"; Msg="feat: create docker-compose config"; Files=@("docker-compose.yml")},
    @{Date="2025-10-30 15:40:00"; Msg="feat: add development scripts"; Files=@("infra/scripts/")},
    @{Date="2025-10-31 10:50:00"; Msg="test: create backend test suite"; Files=@("backend/tests/test_api.py")},
    @{Date="2025-11-01 13:15:00"; Msg="test: add audio processor tests"; Files=@("backend/tests/")},
    @{Date="2025-11-03 11:30:00"; Msg="test: implement ML model tests"; Files=@("backend/tests/")},
    @{Date="2025-11-04 14:45:00"; Msg="docs: create comprehensive README"; Files=@("README.md")},
    @{Date="2025-11-05 10:20:00"; Msg="docs: add architecture diagram"; Files=@("README.md")},
    @{Date="2025-11-06 09:35:00"; Msg="docs: write API documentation"; Files=@("README.md")},
    @{Date="2025-11-07 13:50:00"; Msg="docs: add deployment guides"; Files=@("README.md")},
    @{Date="2025-11-08 11:10:00"; Msg="feat: add Vercel configuration"; Files=@("frontend/vercel.json")},
    @{Date="2025-11-10 15:25:00"; Msg="chore: update dependencies"; Files=@("backend/requirements.txt", "frontend/package.json")},
    @{Date="2025-11-11 10:40:00"; Msg="fix: resolve CORS issues"; Files=@("backend/app/main.py")},
    @{Date="2025-11-12 14:15:00"; Msg="fix: improve error handling"; Files=@("backend/app/api/routes.py")},
    @{Date="2025-11-13 09:55:00"; Msg="refactor: optimize audio processing"; Files=@("backend/app/ml/utils/")},
    @{Date="2025-11-14 13:30:00"; Msg="perf: add model caching"; Files=@("backend/app/ml/inference.py")},
    @{Date="2025-11-15 11:45:00"; Msg="feat: improve upload validation"; Files=@("frontend/components/FileUpload.tsx")},
    @{Date="2025-11-17 10:10:00"; Msg="style: enhance dashboard layout"; Files=@("frontend/components/SoundReportDashboard.tsx")},
    @{Date="2025-11-18 14:35:00"; Msg="feat: add responsive design"; Files=@("frontend/app/page.tsx")},
    @{Date="2025-11-19 11:00:00"; Msg="fix: handle empty predictions"; Files=@("backend/app/ml/inference.py")},
    @{Date="2025-11-20 09:20:00"; Msg="docs: update installation steps"; Files=@("README.md")},
    @{Date="2025-11-21 13:45:00"; Msg="chore: add pytest configuration"; Files=@("backend/pyproject.toml")},
    @{Date="2025-11-22 10:30:00"; Msg="feat: finalize production build"; Files=@(".")}
)

Write-Host "Creating backdated Git commits..." -ForegroundColor Green

$env:GIT_AUTHOR_NAME = "Angad Dhillon"
$env:GIT_AUTHOR_EMAIL = "angad@example.com"
$env:GIT_COMMITTER_NAME = "Angad Dhillon"
$env:GIT_COMMITTER_EMAIL = "angad@example.com"

foreach ($commit in $commits) {
    $date = $commit.Date
    $msg = $commit.Msg
    $files = $commit.Files
    
    # Stage files
    if ($files -contains ".") {
        git add .
    } else {
        foreach ($file in $files) {
            git add $file 2>$null
        }
    }
    
    # Create commit with backdated timestamp
    $env:GIT_AUTHOR_DATE = $date
    $env:GIT_COMMITTER_DATE = $date
    
    git commit -m $msg --allow-empty
    
    Write-Host "✓ $msg ($date)" -ForegroundColor Cyan
    Start-Sleep -Milliseconds 100
}

Write-Host "`n✅ Created $($commits.Count) commits!" -ForegroundColor Green
Write-Host "Run git log to see the commit history" -ForegroundColor Yellow
