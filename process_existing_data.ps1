# PowerShell脚本：并行处理已生成的对局数据
# 功能：适配task子目录结构，并行处理match_data/task_*下的对局数据
# 默认使用连续三回合的结果（history_length=2）

# ========== param块必须放在脚本最顶部 ==========
param(
    [string]$matchDir,
    [string]$behaviorDir,
    [string]$valueDir,
    [int]$historyLength = 2,  # 默认使用连续三回合的结果
    [int]$parallelTasks = 4,  # 默认并行任务数（匹配task_0~task_3共4个目录）
    [switch]$verbose
)

# 获取脚本文件所在的绝对路径（核心：基于脚本位置解析路径）
$scriptDir = $PSScriptRoot
Write-Output "脚本所在目录: $scriptDir"

# 设置默认路径（基于脚本目录，适配task子目录结构）
if (-not $matchDir) { $matchDir = Join-Path -Path $scriptDir -ChildPath "match_data" }
if (-not $behaviorDir) { $behaviorDir = Join-Path -Path $scriptDir -ChildPath "training_data/behavior" }
if (-not $valueDir) { $valueDir = Join-Path -Path $scriptDir -ChildPath "training_data/value" }

# 设置UTF-8编码
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

Write-Output "=== 台球AI训练数据并行处理系统 ==="
Write-Output "此脚本将并行处理match_data/task_*子目录下的对局数据"
Write-Output ""

# 确保路径为绝对路径
$matchDir = [System.IO.Path]::GetFullPath($matchDir)
$behaviorDir = [System.IO.Path]::GetFullPath($behaviorDir)
$valueDir = [System.IO.Path]::GetFullPath($valueDir)

Write-Output "配置信息："
Write-Output "- 对局数据根目录: $matchDir"
Write-Output "- 行为网络数据目录: $behaviorDir"
Write-Output "- 价值网络数据目录: $valueDir"
Write-Output "- 历史回合数: $historyLength (使用连续 $($historyLength + 1) 回合的结果)"
Write-Output "- 并行任务数: $parallelTasks"
Write-Output ""

# 检查对局数据根目录是否存在
if (-not (Test-Path $matchDir -PathType Container)) {
    Write-Output "错误: 对局数据根目录不存在: $matchDir"
    exit 1
}

# 扫描match_data下的task_*子目录（核心修复：适配子目录结构）
Write-Output "正在扫描对局数据子目录（task_*）..."
$taskDirs = Get-ChildItem -Path $matchDir -Directory -Filter "task_*" | 
            Where-Object { $_.Name -match 'task_(\d+)' } | 
            Sort-Object { [int]($_.Name -replace 'task_', '') }

$totalTaskDirs = $taskDirs.Count

if ($totalTaskDirs -eq 0) {
    Write-Output "错误: 未找到任何task_*子目录（目录：$matchDir）"
    exit 1
}

Write-Output "找到 $totalTaskDirs 个task子目录: $($taskDirs.Name -join ', ')"

# 验证并行任务数不超过实际task目录数
if ($parallelTasks -gt $totalTaskDirs) {
    Write-Output "提示：并行任务数($parallelTasks)超过实际task目录数($totalTaskDirs)，自动调整为 $totalTaskDirs"
    $parallelTasks = $totalTaskDirs
}

# 创建输出目录
Write-Output "创建输出目录..."
New-Item -ItemType Directory -Force -Path $behaviorDir | Out-Null
New-Item -ItemType Directory -Force -Path $valueDir | Out-Null

# 创建任务脚本块（适配子目录处理逻辑）
$taskScriptBlock = {
    param(
        $TaskDir,          # 单个task子目录（如match_data/task_0）
        $BehaviorDir,
        $ValueDir,
        $HistoryLength,
        $TaskId,
        $Verbose
    )
    
    Write-Output "任务 ${TaskId} 开始处理子目录: $TaskDir"
    
    # 检查当前task目录下是否有对局文件
    $matchFiles = Get-ChildItem -Path $TaskDir -Filter "match_*.json" -File
    if ($matchFiles.Count -eq 0) {
        Write-Output "任务 ${TaskId} 警告：子目录 $TaskDir 下未找到任何match_*.json文件"
        return $true  # 无文件视为成功，避免任务失败
    }
    Write-Output "任务 ${TaskId} 找到 $($matchFiles.Count) 个对局文件"

    # 构建Python命令参数（适配子目录，传递单个task目录）
    $pythonArgs = @(
        "process_match_data.py",
        "--match_dir", $TaskDir,  # 传递单个task子目录，而非根目录
        "--behavior_output_dir", $BehaviorDir,
        "--value_output_dir", $ValueDir,
        "--history_length", $HistoryLength
    )
    
    if ($Verbose) {
        $pythonArgs += "--verbose"
    }
    
    # 执行Python脚本
    $startTime = Get-Date
    try {
        # 检查Python脚本是否存在
        $pythonScriptPath = Join-Path -Path $PSScriptRoot -ChildPath "process_match_data.py"
        if (-not (Test-Path $pythonScriptPath -File)) {
            Write-Output "任务 ${TaskId} 错误：未找到process_match_data.py（路径：$pythonScriptPath）"
            return $false
        }
        $pythonArgs[0] = $pythonScriptPath

        # 日志文件保存到task目录
        $stdoutLog = Join-Path -Path $TaskDir -ChildPath "task_${TaskId}_output.log"
        $stderrLog = Join-Path -Path $TaskDir -ChildPath "task_${TaskId}_error.log"
        
        $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog
        $process.WaitForExit()
        
        $exitCode = $process.ExitCode
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-Output "任务 ${TaskId} 成功完成，耗时: $duration 秒"
            return $true
        } else {
            Write-Output "任务 ${TaskId} 失败，退出码: $exitCode（查看日志：$stderrLog）"
            return $false
        }
    } catch {
        Write-Output "任务 ${TaskId} 执行出错: $_"
        return $false
    }
}

# 启动并行任务（按task子目录分配任务）
Write-Output "开始并行处理task子目录..."
Write-Output "=" * 50

$tasks = @()
$successCount = 0
$failCount = 0

# 遍历每个task子目录，分配任务
for ($i = 0; $i -lt $parallelTasks; $i++) {
    $currentTaskDir = $taskDirs[$i]
    $taskId = $currentTaskDir.Name -replace 'task_', ''  # 从目录名提取taskId（0/1/2/3）
    
    # 启动后台任务，传递单个task子目录路径
    $task = Start-Job -ScriptBlock $taskScriptBlock -ArgumentList `
        $currentTaskDir.FullName, $behaviorDir, $valueDir, $historyLength, $taskId, $verbose
    $tasks += $task
    
    Write-Output "启动任务 ${taskId}: 处理子目录 $($currentTaskDir.Name)"
}

# 监控任务进度
Write-Output "=" * 50
Write-Output "监控任务进度..."

$allTasksCompleted = $false
while (-not $allTasksCompleted) {
    Start-Sleep -Seconds 2
    
    $runningTasks = Get-Job -Id $tasks.Id | Where-Object { $_.State -eq "Running" }
    $completedTasks = Get-Job -Id $tasks.Id | Where-Object { $_.State -eq "Completed" }
    
    # 处理已完成的任务
    foreach ($task in $completedTasks) {
        if (-not $task.HasMoreData) {
            continue
        }
        
        $result = Receive-Job -Job $task
        if ($result -eq $true) {
            $successCount++
        } else {
            $failCount++
        }
    }
    
    $totalCompleted = $successCount + $failCount
    Write-Output "进度: ${totalCompleted}/${tasks.Count} 任务完成 (${successCount} 成功, ${failCount} 失败)"
    
    # 检查是否所有任务都已完成
    if ($runningTasks.Count -eq 0) {
        $allTasksCompleted = $true
    }
}

Write-Output "=" * 50
Write-Output "所有任务已完成！"
Write-Output "成功: $successCount, 失败: $failCount"

# 清理任务
foreach ($task in $tasks) {
    Remove-Job -Job $task -Force
}

# 总结
Write-Output ""
Write-Output "=== 数据处理完成 ==="
Write-Output "- 行为网络数据保存至: $behaviorDir"
Write-Output "- 价值网络数据保存至: $valueDir"
Write-Output "- 历史回合数: $historyLength (连续 $($historyLength + 1) 回合)"
Write-Output ""

if ($failCount -eq 0) {
    Write-Output "✅ 所有任务执行成功！"
    exit 0
} else {
    Write-Output "❌ 部分任务执行失败，请检查各task目录下的error.log日志。"
    exit 1
}