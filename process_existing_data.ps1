<#
.SYNOPSIS
四并行处理数据（解决文件争抢+路径隔离+无任何报错）
.DESCRIPTION
每个任务写入独立的task_*子目录，彻底避免多进程文件争抢，保留实时控制台输出
#>

# ========== 核心配置 ==========
$scriptWorkingDir = $PSScriptRoot  # 脚本所在绝对路径
$pythonScriptFileName = "process_match_data.py"
$historyLength = 2
$TOTAL_TASKS = 4

# ========== 路径验证与创建（关键：每个任务独立目录）==========
$pythonScriptPath = Join-Path -Path $scriptWorkingDir -ChildPath $pythonScriptFileName
if (-not (Test-Path -Path $pythonScriptPath -PathType Leaf)) {
    Write-Host "❌ 错误：未找到Python脚本 - ${pythonScriptPath}" -ForegroundColor Red
    exit 1
}

# 扫描task目录（仅保留绝对路径）
$taskDirs = @()
for ($i=0; $i -lt $TOTAL_TASKS; $i++) {
    $taskDir = Join-Path -Path $scriptWorkingDir -ChildPath "match_data/task_${i}"
    if (Test-Path $taskDir) {
        # 为每个任务创建独立的输出子目录（关键：避免文件争抢）
        $behaviorTaskDir = Join-Path -Path $scriptWorkingDir -ChildPath "training_data/behavior/task_${i}"
        $valueTaskDir = Join-Path -Path $scriptWorkingDir -ChildPath "training_data/value/task_${i}"
        New-Item -Path $behaviorTaskDir, $valueTaskDir -ItemType Directory -Force | Out-Null
        
        # 存储当前任务的所有路径（输入+输出）
        $taskDirs += @{
            InputDir    = $taskDir
            BehaviorDir = $behaviorTaskDir
            ValueDir    = $valueTaskDir
            TaskId      = $i
        }
    } else {
        Write-Host "⚠️ 警告：task_${i}目录不存在，跳过" -ForegroundColor Yellow
    }
}

if ($taskDirs.Count -eq 0) {
    Write-Host "❌ 错误：无可用的task目录" -ForegroundColor Red
    exit 1
}

# ========== 任务脚本块（核心：每个任务写入独立目录）==========
$taskScriptBlock = {
    param(
        [int]$TASK_ID,
        [string]$pythonScriptPath,
        [string]$matchDir,
        [string]$behaviorDir,
        [string]$valueDir,
        [int]$historyLength
    )

    # 1. 仅保留核心编码配置
    $env:PYTHONIOENCODING = "UTF-8"
    $env:PYTHONLEGACYWINDOWSSTDIO = "UTF-8"
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

    # 2. 输出启动信息（仅控制台）
    Write-Output "`n====================================="
    Write-Output "[$(Get-Date -Format HH:mm:ss)] 任务${TASK_ID} 开始执行"
    Write-Output "输入目录：${matchDir}"
    Write-Output "输出目录：behavior=${behaviorDir} | value=${valueDir}"
    Write-Output "Python脚本：${pythonScriptPath}"
    Write-Output "====================================="

    try {
        # 3. 直接调用Python（每个任务写入独立目录，无文件争抢）
        & python.exe $pythonScriptPath `
            --match_dir $matchDir `
            --behavior_output_dir $behaviorDir `
            --value_output_dir $valueDir `
            --history_length $historyLength

        $exitCode = $LASTEXITCODE

        # 4. 验证JSON文件生成
        $jsonFiles = Get-ChildItem -Path $behaviorDir, $valueDir -Filter "*.json" -ErrorAction SilentlyContinue | 
                     Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) }

        if ($exitCode -eq 0 -and $jsonFiles) {
            $successMsg = "✅ 任务${TASK_ID} 执行成功！生成JSON文件数：$($jsonFiles.Count)，存储路径：${behaviorDir} 和 ${valueDir}"
            Write-Output $successMsg
            return @{ TaskId = $TASK_ID; Success = $true; Message = $successMsg }
        } elseif ($exitCode -eq 0 -and -not $jsonFiles) {
            $errorMsg = "❌ 任务${TASK_ID} 退出码0，但未生成JSON文件（路径：${behaviorDir}）"
            Write-Output $errorMsg
            return @{ TaskId = $TASK_ID; Success = $false; Message = $errorMsg }
        } else {
            $errorMsg = "❌ 任务${TASK_ID} 执行失败！退出码：$exitCode"
            Write-Output $errorMsg
            return @{ TaskId = $TASK_ID; Success = $false; Message = $errorMsg }
        }
    }
    catch {
        $errorMsg = "❌ 任务${TASK_ID} 执行异常：$($_.Exception.Message)"
        Write-Output $errorMsg
        return @{ TaskId = $TASK_ID; Success = $false; Message = $errorMsg }
    }
}

# ========== 启动并行任务 ==========
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "启动$($taskDirs.Count)个并行任务（路径隔离）" -ForegroundColor Cyan
Write-Host "工作目录：${scriptWorkingDir}" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

$tasks = @()
foreach ($taskInfo in $taskDirs) {
    $job = Start-Job -ScriptBlock $taskScriptBlock -ArgumentList `
        $taskInfo.TaskId, $pythonScriptPath, $taskInfo.InputDir, $taskInfo.BehaviorDir, $taskInfo.ValueDir, $historyLength
    $tasks += $job
    Write-Host "✅ 已启动任务$($taskInfo.TaskId)（作业ID：$($job.Id)），输出目录：training_data/behavior/task_$($taskInfo.TaskId)" -ForegroundColor Green
}

# ========== 实时输出所有任务的控制台日志 ==========
Write-Host "`n📢 实时输出任务执行日志（按Ctrl+C可终止）：`n" -ForegroundColor Cyan
do {
    foreach ($task in $tasks) {
        if ($task.HasMoreData) {
            $output = Receive-Job -Job $task
            if ($output) {
                Write-Host $output
            }
        }
    }
    Start-Sleep -Milliseconds 300
} while ($tasks | Where-Object { $_ -is [System.Management.Automation.Job] -and $_.State -notin "Completed", "Failed", "Stopped" })

# ========== 汇总结果 ==========
Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "📊 任务执行汇总" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

$successCount = 0
$failedCount = 0
$results = @()

foreach ($task in $tasks) {
    # 获取任务结果
    $taskOutput = Receive-Job -Job $task
    $result = $taskOutput | Where-Object { $_ -is [hashtable] }
    
    if ($result) {
        $results += $result
        if ($result.Success) {
            $successCount++
            Write-Host "任务$result.TaskId：$($result.Message)" -ForegroundColor Green
        } else {
            $failedCount++
            Write-Host "任务$result.TaskId：$($result.Message)" -ForegroundColor Red
        }
    }
    
    # 清理后台作业
    Remove-Job -Job $task -Force
}

# 最终统计
Write-Host "`n📈 总计：成功${successCount}个 | 失败${failedCount}个" -ForegroundColor Cyan
if ($failedCount -eq 0) {
    Write-Host "🎉 所有任务执行成功！JSON文件已分别存储至：" -ForegroundColor Green
    Write-Host "  - training_data/behavior/task_0 ~ task_3" -ForegroundColor Green
    Write-Host "  - training_data/value/task_0 ~ task_3" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ 存在失败任务，请检查控制台日志排查问题" -ForegroundColor Red
    exit 1
}