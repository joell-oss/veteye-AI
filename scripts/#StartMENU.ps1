<#
   ========================================================================================================================
   Name         : StartMENU.ps1
   Description  : Menu wyboru opcji uruchomienia systemu wykrywania ci¹¿y u klaczy
   Created Date : 2025-05-20
   Created By   : 67193-CKP Józef Sroka
   Dependencies : 1) Windows PowerShell 5.1
                  2) zainstalowane œrodowisko [Anaconda](https://www.anaconda.com/) lub [Python](https://www.python.org/)
                  3) aktywne œrodowisko conda o nazwie `veteye2_env` (lub inne, zgodnie z konfiguracj¹)
                  4) uprawnienia do uruchamiania skryptów PowerShell (`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`)

   Revision History
   Date       Release  Change By                 Description
   2025-05-20 1.0      67193-CKP Józef Sroka     pierwsze wydanie
   ========================================================================================================================
#>



# bie¿¹cy katalog jako katalog scripts
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
Set-Location $scriptDir

# scie¿ka do g³ównego katalogu projektu (poziom wy¿ej)
$projectDir = Split-Path -Parent $scriptDir

function Show-Menu {
    Clear-Host
    Write-Host "======veteye.AI- system predykcji ci¹¿y klaczy na podstawie diagnostyki obrazowej USG -ALK.BIZNES.AI.G12.G2, 2025-MENU======" -ForegroundColor Cyan
    Write-Host
    Write-Host "1: Pe³ny proces (trening modeli i analiza)" -ForegroundColor Green
    Write-Host "2: Tylko analiza obrazów (u¿ywa gotowych modeli)" -ForegroundColor Green
    Write-Host "3: Uruchom interfejs graficzny" -ForegroundColor Green
    Write-Host "4: Analiza pojedynczego obrazu" -ForegroundColor Green
    Write-Host "5: Przetwarzanie wsadowe wielu obrazów" -ForegroundColor Green
    Write-Host "6: Ewaluacja istniej¹cego modelu" -ForegroundColor Green
    Write-Host "7: Wznowienie treningu modelu" -ForegroundColor Green
    Write-Host
    Write-Host "8: Uruchom interfejs graficzny (WEB)" -ForegroundColor Green
    Write-Host
    Write-Host "Q: Wyjœcie" -ForegroundColor Red
    Write-Host
    Write-Host "============================================================================================================================" -ForegroundColor Cyan
}

function Start-Training {
    Write-Host "Uruchamianie treningu modelu wykrywania ci¹¿y..." -ForegroundColor Yellow
    conda activate D:/python/veteye2/veteye2_env
    python main.py --train --model-type pregnancy --train-dir "$projectDir/USG-Mares-Pregnancy-Dataset"
    
    $trainDays = Read-Host "Czy chcesz równie¿ trenowaæ model szacowania dni ci¹¿y? (T/N)"
    if ($trainDays -eq "T" -or $trainDays -eq "t") {
        Write-Host "Uruchamianie treningu modelu szacowania dni ci¹¿y..." -ForegroundColor Yellow
        python main.py --train --model-type day --train-dir "$projectDir/USG-Mares-Pregnancy-Days"
    }
}

function Start-GUI {
    Write-Host "Uruchamianie interfejsu graficznego..." -ForegroundColor Yellow
    conda activate D:/python/veteye2/veteye2_env
    python main.py --analyze
}

function Start-WebGUI {
    Write-Host "Uruchamianie interfejsu przegl¹darkowego (WEB GUI)..." -ForegroundColor Yellow

    $envPath = "D:/python/veteye2/veteye2_env"
    $script = @"
conda activate `"$envPath`"
python web_gui.py
"@

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $script
}


function Analyze-SingleImage {
    Write-Host "Analiza pojedynczego obrazu" -ForegroundColor Yellow
    $imagePath = Read-Host "Podaj œcie¿kê do obrazu do analizy"
    if (Test-Path $imagePath) {
        conda activate D:/python/veteye2/veteye2_env
        python main.py --analyze --image $imagePath
    } else {
        Write-Host "B³¹d: Podany plik nie istnieje!" -ForegroundColor Red
    }
}

function Process-BatchImages {
    Write-Host "Przetwarzanie wsadowe wielu obrazów" -ForegroundColor Yellow
    $inputDir = Read-Host "Podaj œcie¿kê do katalogu z obrazami"
    if (Test-Path $inputDir) {
        $generateReport = Read-Host "Czy generowaæ raport zbiorczy? (T/N)"
        
        conda activate D:/python/veteye2/veteye2_env
        if ($generateReport -eq "T" -or $generateReport -eq "t") {
            python main.py --batch --input-dir $inputDir --report
        } else {
            python main.py --batch --input-dir $inputDir
        }
    } else {
        Write-Host "B³¹d: Podany katalog nie istnieje!" -ForegroundColor Red
    }
}

function Evaluate-Model {
    Write-Host "Ewaluacja istniej¹cego modelu" -ForegroundColor Yellow
    
    # Pobierz listê dostêpnych modeli
    $modelsDir = "$projectDir/checkpoints"
    if (Test-Path $modelsDir) {
        $models = Get-ChildItem -Path $modelsDir -Filter "*.keras"
        
        if ($models.Count -eq 0) {
            Write-Host "Nie znaleziono ¿adnych modeli w katalogu $modelsDir" -ForegroundColor Red
            return
        }
        
        Write-Host "Dostêpne modele:" -ForegroundColor Cyan
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
            Write-Host "Nieprawid³owy wybór modelu!" -ForegroundColor Red
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
        Write-Host "Nieprawid³owy typ modelu! Dozwolone wartoœci: pregnancy, day" -ForegroundColor Red
    }
}

function Full-Process {
    Write-Host "Uruchamianie pe³nego procesu treningu i analizy..." -ForegroundColor Yellow
    
    # 1. Trenowanie modelu wykrywania ci¹¿y
    Write-Host "1. Trenowanie modelu wykrywania ci¹¿y..." -ForegroundColor Yellow
    conda activate D:/python/veteye2/veteye2_env
    python main.py --train --model-type pregnancy --train-dir "$projectDir/USG-Mares-Pregnancy-Dataset"
    
    # 2. Trenowanie modelu szacowania dni ci¹¿y (opcjonalnie)
    $trainDays = Read-Host "Czy chcesz trenowaæ model szacowania dni ci¹¿y? (T/N)"
    if ($trainDays -eq "T" -or $trainDays -eq "t") {
        Write-Host "2. Trenowanie modelu szacowania dni ci¹¿y..." -ForegroundColor Yellow
        python main.py --train --model-type day --train-dir "$projectDir/USG-Mares-Pregnancy-Days"
    }
    
    # 3. Uruchomienie interfejsu graficznego
    $startGUI = Read-Host "Czy chcesz uruchomiæ interfejs graficzny do analizy? (T/N)"
    if ($startGUI -eq "T" -or $startGUI -eq "t") {
        Write-Host "3. Uruchamianie interfejsu graficznego..." -ForegroundColor Yellow
        python main.py --analyze
    }
}

function Analysis-Only {
    Write-Host "Uruchamianie analizy z gotowymi modelami..." -ForegroundColor Yellow
    
    # Wybór trybu analizy
    Write-Host "Wybierz tryb analizy:" -ForegroundColor Cyan
    Write-Host "1: Interfejs graficzny" -ForegroundColor Green
    Write-Host "2: Analiza pojedynczego obrazu" -ForegroundColor Green
    Write-Host "3: Przetwarzanie wsadowe wielu obrazów" -ForegroundColor Green
    
    $analysisMode = Read-Host "Wybierz opcjê (1-3)"
    
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
            Write-Host "Nieprawid³owy wybór!" -ForegroundColor Red
        }
    }
}

# G³ówna pêtla menu
do {
    Show-Menu
    $selection = Read-Host "Wybierz opcjê"
    
    switch ($selection) {
        '1' {
            Full-Process
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '2' {
            Analysis-Only
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '3' {
            Start-GUI
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '4' {
            Analyze-SingleImage
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '5' {
            Process-BatchImages
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '6' {
            Evaluate-Model
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '7' {
            Resume-Training
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        }
        '8' {
            Start-WebGUI
            Write-Host "Naciœnij dowolny klawisz, aby kontynuowaæ..."
            $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
}

    }
} until ($selection -eq 'q' -or $selection -eq 'Q')
cd ..
clear
