# PowerShell脚本：并行处理已生成的对局数据
# 功能：使用多线程并行处理现有的对局数据，生成包含历史状态的训练数据
# 默认使用连续三回合的结果（history_length=2）

# ========== 关键修复：param块必须放在脚本最顶部（注释后、其他代码前） ==========
# 解析命令行参数（必须放在最前面）
param(
    [string]$matchDir,
    [string]$behaviorDir,
    [string]$valueDir,
    [int]$historyLength = 2,  # 默认使用连续三回合的结果（0表示仅当前回合，1表示当前+1个历史，2表示当前+2个历史）
    [int]$parallelTasks = 4,  # 默认并行任务数
    [switch]$verbose
)

# 获取脚本文件所在的绝对路径（核心修复：基于脚本位置解析相对路径）
$scriptDir = $PSScriptRoot  # PowerShell内置变量，指向脚本所在目录
Write-Output "脚本所在目录: $scriptDir"

# 设置默认路径（基于脚本目录，而非当前工作目录）
if (-not $matchDir) { $matchDir = Join-Path -Path $scriptDir -ChildPath "match_data" }
if (-not $behaviorDir) { $behaviorDir = Join-Path -Path $scriptDir -ChildPath "training_data/behavior" }
if (-not $valueDir) { $valueDir = Join-Path -Path $scriptDir -ChildPath "training_data/value" }

# 设置UTF-8编码
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

Write-Output "=== 台球AI训练数据并行处理系统 ==="
Write-Output "此脚本将并行处理已生成的对局数据，默认使用连续三回合的结果"
Write-Output ""

# 确保路径为绝对路径（基于脚本目录解析）
$matchDir = [System.IO.Path]::GetFullPath($matchDir)
$behaviorDir = [System.IO.Path]::GetFullPath($behaviorDir)
$valueDir = [System.IO.Path]::GetFullPath($valueDir)

Write-Output "配置信息："
Write-Output "- 对局数据目录: $matchDir"
Write-Output "- 行为网络数据目录: $behaviorDir"
Write-Output "- 价值网络数据目录: $valueDir"
Write-Output "- 历史回合数: $historyLength (使用连续 $($historyLength + 1) 回合的结果)"
Write-Output "- 并行任务数: $parallelTasks"
Write-Output ""

# 检查对局数据目录是否存在
if (-not (Test-Path $matchDir -PathType Container)) {
    Write-Output "错误: 对局数据目录不存在: $matchDir"
    Write-Output "提示：请确认 match_data 文件夹在脚本目录 [$scriptDir] 下，或通过 -matchDir 参数指定正确路径"
    exit 1
}

# 创建输出目录
Write-Output "创建输出目录..."
New-Item -ItemType Directory -Force -Path $behaviorDir | Out-Null
New-Item -ItemType Directory -Force -Path $valueDir | Out-Null

# 获取所有对局文件并按ID排序
Write-Output "正在扫描对局数据文件..."
$matchFiles = Get-ChildItem -Path $matchDir -Filter "match_*.json" | 
                Where-Object { $_.Name -match 'match_(\d+)\.json' } | 
                Sort-Object { [int]($_.Name -replace '\D', '') }

$totalMatches = $matchFiles.Count

if ($totalMatches -eq 0) {
    Write-Output "错误: 未找到任何对局数据文件（目录：$matchDir）"
    exit 1
}

Write-Output "找到 $totalMatches 个对局数据文件"

# 计算每个任务处理的对局数量
$matchesPerTask = [math]::Ceiling($totalMatches / $parallelTasks)
Write-Output "每个任务处理约 $matchesPerTask 个对局"
Write-Output ""

# 创建任务脚本块
$taskScriptBlock = {
    param(
        $MatchDir,
        $BehaviorDir,
        $ValueDir,
        $StartId,
        $EndId,
        $HistoryLength,
        $TaskId,
        $Verbose
    )
    
    # 使用 ${变量名} 明确分隔变量，避免语法歧义
    Write-Output "任务 ${TaskId} 开始处理对局ID范围: ${StartId} - ${EndId}"
    
    # 构建Python命令参数
    $pythonArgs = @(
        "process_match_data.py",
        "--match_dir", $MatchDir,
        "--behavior_output_dir", $BehaviorDir,
        "--value_output_dir", $ValueDir,
        "--start_id", $StartId,
        "--end_id", $EndId,
        "--history_length", $HistoryLength
    )
    
    if ($Verbose) {
        $pythonArgs += "--verbose"
    }
    
    # 执行Python脚本（确保Python脚本路径基于任务目录）
    $pythonScriptPath = Join-Path -Path $PWD.Path -ChildPath "process_match_data.py"
    if (-not (Test-Path $pythonScriptPath)) {
        Write-Output "任务 ${TaskId} 错误：未找到 process_match_data.py（路径：$pythonScriptPath）"
        return $false
    }
    $pythonArgs[0] = $pythonScriptPath

    # 执行Python脚本
    $startTime = Get-Date
    try {
        # 日志文件名中的 ${TaskId} 明确分隔，日志保存到脚本目录
        $logDir = $MatchDir | Split-Path -Parent
        $stdoutLog = Join-Path -Path $logDir -ChildPath "./task_${TaskId}_output.log"
        $stderrLog = Join-Path -Path $logDir -ChildPath "./task_${TaskId}_error.log"
        
        $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog
        $process.WaitForExit()
        
        $exitCode = $process.ExitCode
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        # ${TaskId} 明确分隔
        if ($exitCode -eq 0) {
            Write-Output "任务 ${TaskId} 成功完成，耗时: $duration 秒"
            return $true
        } else {
            Write-Output "任务 ${TaskId} 失败，退出码: $exitCode"
            return $false
        }
    } catch {
        # ${TaskId} 明确分隔
        Write-Output "任务 ${TaskId} 执行出错: $_"
        return $false
    }
}

# 启动并行任务
Write-Output "开始并行处理对局数据..."
Write-Output "=" * 50

$tasks = @()
$successCount = 0
$failCount = 0

for ($taskId = 1; $taskId -le $parallelTasks; $taskId++) {
    $startIndex = ($taskId - 1) * $matchesPerTask
    $endIndex = [math]::Min(($startIndex + $matchesPerTask - 1), ($totalMatches - 1))
    
    # 如果没有对局需要处理，跳过
    if ($startIndex -ge $totalMatches) {
        continue
    }
    
    # 获取对应的对局ID
    $startId = [int]($matchFiles[$startIndex].Name -replace '\D', '')
    $endId = [int]($matchFiles[$endIndex].Name -replace '\D', '')
    
    # 启动后台任务
    $task = Start-Job -ScriptBlock $taskScriptBlock -ArgumentList $matchDir, $behaviorDir, $valueDir, $startId, $endId, $historyLength, $taskId, $verbose
    $tasks += $task
    
    # 使用 ${变量名} 明确分隔变量
    Write-Output "启动任务 ${taskId}: 处理对局ID ${startId} - ${endId}"
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
    # 使用 ${变量名} 明确分隔变量
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
    Write-Output "❌ 部分任务执行失败，请检查错误日志。"
    exit 1
}