<#
   ========================================================================================================================
   Name         : StartMENU.ps1
   Description  : Menu wyboru opcji uruchomienia systemu wykrywania ci��y u klaczy
   Created Date : 2025-05-20
   Created By   : 67193-CKP J�zef Sroka
   Dependencies : 1) Windows PowerShell 5.1
                  2) zainstalowane �rodowisko [Anaconda](https://www.anaconda.com/) lub [Python](https://www.python.org/)
                  3) aktywne �rodowisko conda o nazwie `veteye2_env` (lub inne, zgodnie z konfiguracj�)
                  4) uprawnienia do uruchamiania skrypt�w PowerShell (`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`)

   Revision History
   Date       Release  Change By                 Description
   2025-05-20 1.0      67193-CKP J�zef Sroka     pierwsze wydanie
   ========================================================================================================================
#>



# bie��cy katalog jako katalog scripts
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
Set-Location $scriptDir

# scie�ka do g��wnego katalogu projektu (poziom wy�ej)
$projectDir = Split-Path -Parent $scriptDir

function Show-Menu {
    Clear-Host
    Write-Host "======veteye.AI- system predykcji ci��y klaczy na podstawie diagnostyki obrazowej USG -ALK.BIZNES.AI.G12.G2, 2025-MENU======" -ForegroundColor Cyan
    Write-Host
    Write-Host "1: Pe�ny proces (trening modeli i analiza)" -ForegroundColor Green
    Write-Host "2: Tylko analiza obraz�w (u�ywa gotowych modeli)" -ForegroundColor Green
    Write-Host "3: Uruchom interfejs graficzny" -ForegroundColor Green
    Write-Host "4: Analiza pojedynczego obrazu" -ForegroundColor Green
    Write-Host "5: Przetwarzanie wsadowe wielu obraz�w" -ForegroundColor Green
    Write-Host "6: Ewaluacja istniej�cego modelu" -ForegroundColor Green
    Write-Host "7: Wznowienie treningu modelu" -ForegroundColor Green
    Write-Host
    Write-Host "8: Uruchom interfejs graficzny (WEB)" -ForegroundColor Green
    Write-Host
    Write-Host "Q: Wyj�cie" -ForegroundColor Red
    Write-Host
    Write-Host "============================================================================================================================" -ForegroundColor Cyan
}

function Start-Training {
    Write-Host "Uruchamianie treningu modelu wykrywania ci��y..." -ForegroundColor Yellow
    conda activate D:/python/veteye2/veteye2_env
    python main.py --train --model-type pregnancy --train-dir "$projectDir/USG-Mares-Pregnancy-Dataset"
    
    $trainDays = Read-Host "Czy chcesz r�wnie� trenowa� model szacowania dni ci��y? (T/N)"
    if ($trainDays -eq "T" -or $trainDays -eq "t") {
        Write-Host "Uruchamianie treningu modelu szacowania dni ci��y..." -ForegroundColor Yellow
        python main.py --train --model-type day --train-dir "$projectDir/USG-Mares-Pregnancy-Days"
    }
}

function Start-GUI {
    Write-Host "Uruchamianie interfejsu graficznego..." -ForegroundColor Yellow
    conda activate D:/python/veteye2/veteye2_env
    python main.py --analyze
}

function Start-WebGUI {
    Write-Host "Uruchamianie interfejsu przegl�darkowego (WEB GUI)..." -ForegroundColor Yellow

    $envPath = "D:/python/veteye2/veteye2_env"
    $script = @"
conda activate `"$envPath`"
python web_gui.py
"@

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $script
}


function Analyze-SingleImage {
    Write-Host "Analiza pojedynczego obrazu" -ForegroundColor Yellow
    $imagePath = Read-Host "Podaj �cie�k� do obrazu do analizy"
    if (Test-Path $imagePath) {
        conda activate D:/python/veteye2/veteye2_env
        python main.py --analyze --image $imagePath
    } else {
        Write-Host "B��d: Podany plik nie istnieje!" -ForegroundColor Red
    }
}

function Process-BatchImages {
    Write-Host "Przetwarzanie wsadowe wielu obraz�w" -ForegroundColor Yellow
    $inputDir = Read-Host "Podaj �cie�k� do katalogu z obrazami"
    if (Test-Path $inputDir) {
        $generateReport = Read-Host "Czy generowa� raport zbiorczy? (T/N)"
        
        conda activate D:/python/veteye2/veteye2_env
        if ($generateReport -eq "T" -or $generateReport -eq "t") {
            python main.py --batch --input-dir $inputDir --report
        } else {
            python main.py --batch --input-dir $inputDir
        }
    } else {
        Write-Host "B��d: Podany katalog nie istnieje!" -ForegroundColor Red
    }
}

function Evaluate-Model {
    Write-Host "Ewaluacja istniej�cego modelu" -ForegroundColor Yellow
    
    # Pobierz list� dost�pnych modeli
    $modelsDir = "$projectDir/checkpoints"
    if (Test-Path $modelsDir) {
        $models = Get-ChildItem -Path $modelsDir -Filter "*.keras"
        
        if ($models.Count -eq 0) {
            Write-Host "Nie znaleziono �adnych modeli w katalogu $modelsDir" -ForegroundColor Red
            return
        }
        
        Write-Host "Dost�pne modele:" -ForegroundColor Cyan
        for ($i=0; $i -lt $models.Count; $i++) {
            Write-Host "$($i+1): $($models[$i].Name)" -ForegroundColor Green
        }
        
        $selection = Read-Host "Wybierz numer modelu do ewaluacji (1-$($models.Count))"
        $index = [int]$selection - 1
        
        if ($index -ge 0 -and $index -lt $models.Count) {
            $selectedModel = $models[$index].FullName
            
            conda activate D:/python/veteye2/veteye2_env
            python main.py --train --model-type pregnancy --train-dir "$projectDir/USG-Mares-Pregnancy-Dataset" --resume --model $selectedModel
        } else {
            Write-Host "Nieprawid�owy wyb�r modelu!" -ForegroundColor Red
        }
    } else {
        Write-Host "Katalog modeli ($modelsDir) nie istnieje!" -ForegroundColor Red
    }
}

function Resume-Training {
    Write-Host "Wznowienie treningu modelu" -ForegroundColor Yellow
    
    $modelType = Read-Host "Podaj typ modelu do wznowienia (pregnancy/day)"
    if ($modelType -eq "pregnancy" -or $modelType -eq "day") {
        conda activate D:/python/veteye2/veteye2_env
        python main.py --train --model-type $modelType --train-dir "$projectDir/USG-Mares-Pregnancy-Dataset" --resume
    } else {
        Write-Host "Nieprawid�owy typ modelu! Dozwolone warto�ci: pregnancy, day" -ForegroundColor Red
    }
}

function Full-Process {
    Write-Host "Uruchamianie pe�nego procesu treningu i analizy..." -ForegroundColor Yellow
    
    # 1. Trenowanie modelu wykrywania ci��y
    Write-Host "1. Trenowanie modelu wykrywania ci��y..." -ForegroundColor Yellow
    conda activate D:/python/veteye2/veteye2_env
    python main.py --train --model-type pregnancy --train-dir "$projectDir/USG-Mares-Pregnancy-Dataset"
    
    # 2. Trenowanie modelu szacowania dni ci��y (opcjonalnie)
    $trainDays = Read-Host "Czy chcesz trenowa� model szacowania dni ci��y? (T/N)"
    if ($trainDays -eq "T" -or $trainDays -eq "t") {
        Write-Host "2. Trenowanie modelu szacowania dni ci��y..." -ForegroundColor Yellow
        python main.py --train --model-type day --train-dir "$projectDir/USG-Mares-Pregnancy-Days"
    }
    
    # 3. Uruchomienie interfejsu graficznego
    $startGUI = Read-Host "Czy chcesz uruchomi� interfejs graficzny do analizy? (T/N)"
    if ($startGUI -eq "T" -or $startGUI -eq "t") {
        Write-Host "3. Uruchamianie interfejsu graficznego..." -ForegroundColor Yellow
        python main.py --analyze
    }
}

function Analysis-Only {
    Write-Host "Uruchamianie analizy z gotowymi modelami..." -ForegroundColor Yellow
    
    # Wyb�r trybu analizy
    Write-Host "Wybierz tryb analizy:" -ForegroundColor Cyan
    Write-Host "1: Interfejs graficzny" -ForegroundColor Green
    Write-Host "2: Analiza pojedynczego obrazu" -ForegroundColor Green
    Write-Host "3: Przetwarzanie wsadowe wielu obraz�w" -ForegroundColor Green
    
    $analysisMode = Read-Host "Wybierz opcj� (1-3)"
    
    conda activate D:/python/veteye2/veteye2_env
    
    switch ($analysisMode) {
        "1" {
            Write-Host "Uruchamianie interfejsu graficznego..." -ForegroundColor Yellow
            python main.py --analyze
        }
        "2" {
            Analyze-SingleImage
        }
        "3" {
            Process-BatchImages
        }
        default {
            Write-Host "Nieprawid�owy wyb�r!" -ForegroundColor Red
        }
    }
}

# G��wna p�tla menu
do {
    Show-Menu
    $selection = Read-Host "Wybierz opcj�"
    
    switch ($selection) {
        '1' {
            Full-Process
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '2' {
            Analysis-Only
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '3' {
            Start-GUI
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '4' {
            Analyze-SingleImage
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '5' {
            Process-BatchImages
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '6' {
            Evaluate-Model
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '7' {
            Resume-Training
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '8' {
            Start-WebGUI
            Write-Host "Naci�nij dowolny klawisz, aby kontynuowa�..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
}

    }
} until ($selection -eq 'q' -or $selection -eq 'Q')
cd ..
clear
