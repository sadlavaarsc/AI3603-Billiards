<#
.SYNOPSIS
本地PowerShell四并行执行数据生成任务（兼容PowerShell 5.1，修复目录+编码+只读属性报错）
.DESCRIPTION
移除只读属性赋值，仅保留有效编码配置，解决Unicode特殊字符无法编码及报错问题
#>

# 核心配置：优先使用绝对路径
$scriptWorkingDir = $PSScriptRoot  # 脚本所在目录的绝对路径
$TOTAL_MATCHES = 10000
$TOTAL_TASKS = 4  # 四并行
$pythonScriptFileName = "generate_train_data.py"  # 你的Python脚本文件名
$pythonScriptPath = Join-Path -Path $scriptWorkingDir -ChildPath $pythonScriptFileName

# 验证Python脚本是否存在
if (-not (Test-Path -Path $pythonScriptPath -PathType Leaf)) {
    Write-Host "❌ 错误：未找到Python脚本文件，路径为：${pythonScriptPath}"
    exit 1
}

# 提前创建数据目录
$matchDirRoot = Join-Path -Path $scriptWorkingDir -ChildPath "match_data"
$behaviorDirRoot = Join-Path -Path $scriptWorkingDir -ChildPath "training_data/behavior"
$valueDirRoot = Join-Path -Path $scriptWorkingDir -ChildPath "training_data/value"
New-Item -Path $matchDirRoot, $behaviorDirRoot, $valueDirRoot -ItemType Directory -Force | Out-Null

# 定义并行任务的脚本块（移除只读属性赋值，保留有效编码配置）
$taskScriptBlock = {
    param(
        [int]$TASK_ID,
        [int]$TOTAL_MATCHES,
        [int]$TOTAL_TASKS,
        [string]$pythonScriptPath,
        [string]$workingDir
    )

    # 1. 强制切换后台作业的工作目录
    Set-Location -Path $workingDir -ErrorAction Stop

    # 2. 核心有效修复：设置Python环境变量，强制使用UTF-8编码（解决Unicode编码错误）
    # 这两个环境变量是关键，无需修改只读的Default编码
    $env:PYTHONIOENCODING = "UTF-8"
    $env:PYTHONLEGACYWINDOWSSTDIO = "UTF-8"

    # 3. 有效配置：设置控制台输出编码为UTF-8（兼容Python输出的Unicode字符，无报错）
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

    # 移除 [System.Text.Encoding]::Default = [System.Text.Encoding]::UTF8（只读属性，无需配置）

    # 任务分配逻辑
    $MATCHES_PER_TASK = [math]::Floor($TOTAL_MATCHES / $TOTAL_TASKS)
    $REMAINING_MATCHES = $TOTAL_MATCHES % $TOTAL_TASKS

    if ($TASK_ID -lt $REMAINING_MATCHES) {
        $CURRENT_TASK_MATCHES = $MATCHES_PER_TASK + 1
        $START_ID = $TASK_ID * $CURRENT_TASK_MATCHES
    }
    else {
        $CURRENT_TASK_MATCHES = $MATCHES_PER_TASK
        $START_ID = $REMAINING_MATCHES * ($MATCHES_PER_TASK + 1) + ($TASK_ID - $REMAINING_MATCHES) * $MATCHES_PER_TASK
    }

    # 检查当前任务是否需要执行
    if ($CURRENT_TASK_MATCHES -le 0) {
        Write-Output "任务${TASK_ID}：无需执行（总任务数大于总对局数）"
        return @{
            TaskId     = $TASK_ID
            Success    = $true
            StartId    = $START_ID
            MatchCount = $CURRENT_TASK_MATCHES
            Message    = "无需执行"
        }
    }

    # 输出任务启动信息
    Write-Output "`n任务${TASK_ID}：启动执行，start_id=${START_ID}，num_matches=${CURRENT_TASK_MATCHES}"
    Write-Output "任务${TASK_ID}：当前工作目录：$((Get-Location).Path)"
    Write-Output "任务${TASK_ID}：Python脚本路径：${pythonScriptPath}"

    # 构建Python脚本参数（绝对路径）
    $matchDir = Join-Path -Path $workingDir -ChildPath "match_data/task_${TASK_ID}"
    $behaviorDir = Join-Path -Path $workingDir -ChildPath "training_data/behavior/task_${TASK_ID}"
    $valueDir = Join-Path -Path $workingDir -ChildPath "training_data/value/task_${TASK_ID}"

    $pythonArgs = @(
        $pythonScriptPath,
        "--start_id", $START_ID,
        "--num_matches", $CURRENT_TASK_MATCHES,
        "--match_dir", $matchDir,
        "--behavior_dir", $behaviorDir,
        "--value_dir", $valueDir,
        "--enable_noise",
        "--max_hit_count", 60,
        "--verbose"
    )

    # 执行Python脚本并捕获结果
    try {
        # 直接执行Python，已配置UTF-8编码，特殊字符可正常输出
        & python.exe $pythonArgs
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Output "✅ 任务${TASK_ID}执行完成（start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES}）"
            return @{
                TaskId     = $TASK_ID
                Success    = $true
                StartId    = $START_ID
                MatchCount = $CURRENT_TASK_MATCHES
                ExitCode   = $exitCode
                Message    = "执行成功"
            }
        }
        else {
            Write-Output "❌ 任务${TASK_ID}执行失败（start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES}），退出码：${exitCode}"
            return @{
                TaskId     = $TASK_ID
                Success    = $false
                StartId    = $START_ID
                MatchCount = $CURRENT_TASK_MATCHES
                ExitCode   = $exitCode
                Message    = "执行失败，退出码：${exitCode}"
            }
        }
    }
    catch {
        Write-Output "❌ 任务${TASK_ID}执行异常（start_id=${START_ID}, num_matches=${CURRENT_TASK_MATCHES}），错误信息：$($_.Exception.Message)"
        return @{
            TaskId     = $TASK_ID
            Success    = $false
            StartId    = $START_ID
            MatchCount = $CURRENT_TASK_MATCHES
            ExitCode   = -1
            Message    = "执行异常：$($_.Exception.Message)"
        }
    }
}

# 启动4个并行后台任务
Write-Host "====================================="
Write-Host "开始启动${TOTAL_TASKS}个并行任务，总对局数：${TOTAL_MATCHES}"
Write-Host "脚本工作目录：${scriptWorkingDir}"
Write-Host "Python脚本路径：${pythonScriptPath}"
Write-Host "====================================="

$tasks = @()
for ($taskId = 0; $taskId -lt $TOTAL_TASKS; $taskId++) {
    $job = Start-Job -ScriptBlock $taskScriptBlock -ArgumentList $taskId, $TOTAL_MATCHES, $TOTAL_TASKS, $pythonScriptPath, $scriptWorkingDir
    $tasks += $job
    Write-Host "已启动任务${taskId}（作业ID：$($job.Id)）"
}

# 实时轮询获取任务输出
Write-Host "`n所有并行任务已启动，正在实时输出日志...`n"
do {
    foreach ($task in $tasks) {
        if ($task.HasMoreData) {
            Receive-Job -Job $task | Write-Host
        }
    }
    Start-Sleep -Milliseconds 300
} while ($tasks | Where-Object { $_.State -notin "Completed", "Failed", "Stopped" })

# 获取所有任务最终结果并清理后台作业
$results = @()
foreach ($task in $tasks) {
    $taskOutput = Receive-Job -Job $task
    if ($taskOutput) {
        Write-Host $taskOutput
    }
    $result = $taskOutput | Where-Object { $_ -is [hashtable] }
    if ($result) {
        $results += $result
    }
    Remove-Job -Job $task
}

# 输出汇总结果
Write-Host "`n====================================="
Write-Host "所有任务执行完毕，汇总结果："
Write-Host "====================================="

$successCount = ($results | Where-Object { $_.Success -eq $true }).Count
$failedCount = ($results | Where-Object { $_.Success -eq $false }).Count

foreach ($result in $results) {
    if ($result.Success) {
        Write-Host "任务${result.TaskId}：✅ 成功 | 起始ID：${result.StartId} | 对局数：${result.MatchCount} | 备注：${result.Message}"
    }
    else {
        Write-Host "任务${result.TaskId}：❌ 失败 | 起始ID：${result.StartId} | 对局数：${result.MatchCount} | 错误信息：${result.Message}"
    }
}

Write-Host "`n📊 整体汇总：成功${successCount}个任务，失败${failedCount}个任务"
if ($failedCount -gt 0) {
    Write-Host "❌ 存在失败任务，请检查错误信息后重试"
    exit 1
}
else {
    Write-Host "✅ 所有任务均执行成功！数据生成完成，文件存储在：${scriptWorkingDir}"
    exit 0
}