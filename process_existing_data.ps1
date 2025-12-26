<#
.SYNOPSIS
并行处理台球AI训练数据（兼容PowerShell 2.0+，支持长时间运行）
.DESCRIPTION
处理match_data/task_*下的1万条数据，每条60轮结果，适配1~5小时的线程耗时，超时配置6小时
#>

# ========== 核心参数（稳妥配置）==========
param(
    [string]$matchDir,
    [string]$behaviorDir,
    [string]$valueDir,
    [int]$historyLength = 2,
    [int]$parallelTasks = 4,
    [int]$taskTimeout = 21600,  # 6小时（21600秒），覆盖最长5小时耗时+20%冗余
    [int]$pollInterval = 30,    # 30秒轮询一次进度，降低资源占用
    [switch]$verbose
)

# ========== 兼容层（PowerShell 2.0+全兼容）==========
# 1. 兼容脚本目录获取（PowerShell 2.0无$PSScriptRoot）
if ($PSVersionTable.PSVersion.Major -ge 3) {
    $scriptDir = $PSScriptRoot
} else {
    $scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
}
# 强制转为绝对路径，避免相对路径歧义
$scriptDir = [System.IO.Path]::GetFullPath($scriptDir)

# 2. 兼容编码设置（PowerShell 2.0无UTF8编码直接指定，用.NET类）
$utf8Encoding = New-Object System.Text.UTF8Encoding($false)  # 无BOM的UTF8，兼容所有系统
$OutputEncoding = $utf8Encoding
[Console]::OutputEncoding = $utf8Encoding
[Console]::InputEncoding = $utf8Encoding

# 3. 输出环境信息，便于排查
Write-Output "========== 运行环境信息 =========="
Write-Output "PowerShell版本: $($PSVersionTable.PSVersion)"
Write-Output "脚本目录: $scriptDir"
Write-Output "单任务超时: $taskTimeout 秒（$($taskTimeout/3600) 小时）"
Write-Output "轮询间隔: $pollInterval 秒"
Write-Output "==================================="
Write-Output ""

# ========== 路径初始化（全兼容）==========
# 设置默认路径
if (-not $matchDir) { $matchDir = Join-Path -Path $scriptDir -ChildPath "match_data" }
if (-not $behaviorDir) { $behaviorDir = Join-Path -Path $scriptDir -ChildPath "training_data/behavior" }
if (-not $valueDir) { $valueDir = Join-Path -Path $scriptDir -ChildPath "training_data/value" }

# 转为绝对路径
$matchDir = [System.IO.Path]::GetFullPath($matchDir)
$behaviorDir = [System.IO.Path]::GetFullPath($behaviorDir)
$valueDir = [System.IO.Path]::GetFullPath($valueDir)

# 检查根目录是否存在（PowerShell 2.0兼容写法，不用-PathType）
if (-not (Test-Path $matchDir)) {
    Write-Output "错误: 对局数据根目录不存在: $matchDir"
    exit 1
}
$matchDirItem = Get-Item $matchDir -ErrorAction Stop
if ($matchDirItem.PSIsContainer -eq $false) {
    Write-Output "错误: $matchDir 不是文件夹"
    exit 1
}

# ========== 扫描Task目录（全兼容）==========
Write-Output "正在扫描task子目录..."
# PowerShell 2.0兼容的目录筛选（不用-Directory参数）
$taskDirs = Get-ChildItem -Path $matchDir -ErrorAction Stop | 
            Where-Object { 
                $_.PSIsContainer -eq $true -and 
                $_.Name -match '^task_\d+$'  # 严格匹配task_数字格式
            } | 
            Sort-Object { [int]($_.Name -replace 'task_', '') }

$totalTaskDirs = @($taskDirs).Count  # 兼容空数组的Count属性
if ($totalTaskDirs -eq 0) {
    Write-Output "错误: 未找到task_*子目录"
    exit 1
}
Write-Output "找到 $totalTaskDirs 个task子目录: $($taskDirs.Name -join ', ')"

# 调整并行数（不超过实际task数）
if ($parallelTasks -gt $totalTaskDirs) {
    Write-Output "提示: 并行数($parallelTasks)超过task数($totalTaskDirs)，自动调整为 $totalTaskDirs"
    $parallelTasks = $totalTaskDirs
}

# 创建输出目录（PowerShell 2.0兼容，不用-NewItem -Force）
if (-not (Test-Path $behaviorDir)) {
    New-Item -ItemType Directory -Path $behaviorDir -ErrorAction Stop | Out-Null
}
if (-not (Test-Path $valueDir)) {
    New-Item -ItemType Directory -Path $valueDir -ErrorAction Stop | Out-Null
}

# ========== 检查Python环境（全兼容）==========
Write-Output ""
Write-Output "检查Python环境..."
$pythonExe = $null
$pythonCandidates = @("python.exe", "python3.exe", "py.exe")

# 遍历候选，优先找绝对路径
foreach ($candidate in $pythonCandidates) {
    try {
        $pythonPath = Get-Command $candidate -ErrorAction Stop
        if ($pythonPath -and $pythonPath.Source) {
            $pythonExe = $pythonPath.Source
            break
        }
    } catch {
        continue
    }
}

# 兜底：如果没找到绝对路径，用命令名
if (-not $pythonExe) {
    foreach ($candidate in $pythonCandidates) {
        if (Test-Path "env:PATH" | Where-Object { $_ -match $candidate }) {
            $pythonExe = $candidate
            if ($candidate -eq "py.exe") {
                $pythonExe = "py -3"
            }
            break
        }
    }
}

if (-not $pythonExe) {
    Write-Output "错误: 未找到Python执行程序，请确保Python已添加到系统PATH"
    exit 1
}
Write-Output "使用Python: $pythonExe"

# 检查Python脚本（PowerShell 2.0兼容，不用-File参数）
$pythonScriptPath = Join-Path -Path $scriptDir -ChildPath "process_match_data.py"
if (-not (Test-Path $pythonScriptPath)) {
    Write-Output "错误: 未找到Python脚本: $pythonScriptPath"
    exit 1
}
$pythonScriptItem = Get-Item $pythonScriptPath -ErrorAction Stop
if ($pythonScriptItem.PSIsContainer -eq $true) {
    Write-Output "错误: $pythonScriptPath 是文件夹，不是Python脚本"
    exit 1
}
Write-Output "找到Python脚本: $pythonScriptPath"

# ========== 任务脚本块（核心：长时间运行+全兼容）==========
$taskScriptBlock = {
    param(
        $TaskDir,
        $BehaviorDir,
        $ValueDir,
        $HistoryLength,
        $TaskId,
        $PythonExe,
        $PythonScriptPath,
        $TaskTimeout,
        $PollInterval
    )

    # 1. 初始化日志（PowerShell 2.0兼容，Out-File不用-Path参数）
    $debugLog = Join-Path -Path $TaskDir -ChildPath "task_${TaskId}_debug.log"
    $stderrLog = Join-Path -Path $TaskDir -ChildPath "task_${TaskId}_error.log"
    $stdoutLog = Join-Path -Path $TaskDir -ChildPath "task_${TaskId}_output.log"

    # 写入初始日志（用.NET StreamWriter兼容低版本编码）
    $logWriter = New-Object System.IO.StreamWriter($debugLog, $false, [System.Text.UTF8Encoding]::new($false))
    $logWriter.WriteLine("[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 任务${TaskId}启动，处理目录: $TaskDir")
    $logWriter.WriteLine("[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Python路径: $PythonExe")
    $logWriter.WriteLine("[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 脚本路径: $PythonScriptPath")
    $logWriter.WriteLine("[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 超时时间: $TaskTimeout 秒（$($TaskTimeout/3600) 小时）")
    $logWriter.Flush()
    $logWriter.Close()

    # 2. 扫描数据文件（PowerShell 2.0兼容）
    try {
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 开始扫描数据文件"
        Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 开始扫描数据文件" -Encoding UTF8

        $matchFiles = Get-ChildItem -Path $TaskDir -ErrorAction Stop | 
                      Where-Object { 
                          $_.PSIsContainer -eq $false -and 
                          $_.Name -like "match_*.json" 
                      }
        $fileCount = @($matchFiles).Count  # 兼容空数组

        Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 扫描到 $fileCount 个match_*.json文件" -Encoding UTF8
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 找到 $fileCount 个数据文件"

        if ($fileCount -eq 0) {
            Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 无数据文件，任务结束" -Encoding UTF8
            Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 无数据文件，直接成功"
            return $true
        }

        # 3. 构建Python命令（处理路径空格，兼容所有系统）
        $escapedTaskDir = "`"$TaskDir`""
        $escapedBehaviorDir = "`"$BehaviorDir`""
        $escapedValueDir = "`"$ValueDir`""
        $escapedPythonScript = "`"$PythonScriptPath`""

        $pythonArgs = @(
            $escapedPythonScript,
            "--match_dir", $escapedTaskDir,
            "--behavior_output_dir", $escapedBehaviorDir,
            "--value_output_dir", $escapedValueDir,
            "--history_length", $HistoryLength
        )

        $fullCommand = "$PythonExe $($pythonArgs -join ' ')"
        Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 执行命令: $fullCommand" -Encoding UTF8
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 启动Python进程，命令: $fullCommand"

        # 4. 启动Python进程（带超时，兼容长时间运行）
        $startTime = Get-Date
        $process = $null
        $timeoutOccurred = $false

        try {
            # 启动进程（PowerShell 2.0兼容，不用-RedirectStandardOutput的简写）
            if ($PythonExe -match ' ') {
                $process = Start-Process -FilePath "cmd.exe" `
                    -ArgumentList "/c ""$PythonExe"" $($pythonArgs -join ' ')" `
                    -NoNewWindow `
                    -PassThru `
                    -RedirectStandardOutput $stdoutLog `
                    -RedirectStandardError $stderrLog
            } else {
                $process = Start-Process -FilePath $PythonExe `
                    -ArgumentList $pythonArgs `
                    -NoNewWindow `
                    -PassThru `
                    -RedirectStandardOutput $stdoutLog `
                    -RedirectStandardError $stderrLog
            }

            # 5. 超时等待（循环检查，避免长时间阻塞，兼容PowerShell 2.0）
            $elapsedMs = 0
            $checkIntervalMs = $PollInterval * 1000  # 转为毫秒
            $timeoutMs = $TaskTimeout * 1000

            while (-not $process.HasExited -and $elapsedMs -lt $timeoutMs) {
                Start-Sleep -Milliseconds $checkIntervalMs
                $elapsedMs += $checkIntervalMs

                # 每30分钟记录一次心跳（确认进程还在运行）
                if ($elapsedMs % 1800000 -eq 0) {  # 30分钟=1800000毫秒
                    $elapsedHours = [math]::Round($elapsedMs/3600000, 1)
                    Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 进程已运行 $elapsedHours 小时，仍在执行" -Encoding UTF8
                    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 已运行 $elapsedHours 小时，进程正常"
                }
            }

            # 6. 处理超时/正常退出
            if (-not $process.HasExited) {
                # 超时：终止进程
                $process.Kill()
                $timeoutOccurred = $true
                Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 任务超时（$TaskTimeout 秒），已终止进程" -Encoding UTF8
                Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 超时终止（已运行 $($elapsedMs/3600000) 小时）"
                return $false
            }

            # 7. 处理退出结果
            $endTime = Get-Date
            $durationHours = [math]::Round(($endTime - $startTime).TotalHours, 2)
            $exitCode = $process.ExitCode

            Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Python进程退出，退出码: $exitCode，总耗时: $durationHours 小时" -Encoding UTF8

            if ($exitCode -eq 0) {
                Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 成功完成，总耗时: $durationHours 小时"
                Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 任务成功完成" -Encoding UTF8
                return $true
            } else {
                Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 失败，退出码: $exitCode"
                Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 任务失败，退出码: $exitCode" -Encoding UTF8
                # 读取Python错误日志并写入
                if (Test-Path $stderrLog) {
                    $errorContent = Get-Content $stderrLog -Encoding UTF8 -ErrorAction SilentlyContinue
                    if ($errorContent) {
                        Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Python错误输出: $($errorContent -join '; ')" -Encoding UTF8
                    }
                }
                return $false
            }
        } catch {
            $errorMsg = $_.Exception.Message
            Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 执行出错: $errorMsg"
            Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 执行出错: $errorMsg" -Encoding UTF8
            # 写入错误日志（兼容PowerShell 2.0）
            $errorMsg | Out-File $stderrLog -Encoding UTF8
            return $false
        }
    } catch {
        $errorMsg = $_.Exception.Message
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 任务${TaskId}: 初始化出错: $errorMsg"
        Add-Content $debugLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 初始化出错: $errorMsg" -Encoding UTF8
        $errorMsg | Out-File $stderrLog -Encoding UTF8
        return $false
    }
}

# ========== 启动并行任务（长时间运行优化）==========
Write-Output ""
Write-Output "========== 启动并行任务 =========="
$tasks = @()
$successCount = 0
$failCount = 0
$timeoutCount = 0
$globalStartTime = Get-Date

# 启动每个task
for ($i = 0; $i -lt $parallelTasks; $i++) {
    $currentTaskDir = $taskDirs[$i]
    $taskId = $currentTaskDir.Name -replace 'task_', ''

    # 启动Job（PowerShell 2.0兼容，不用-ArgumentList的简写）
    $task = Start-Job -ScriptBlock $taskScriptBlock -ArgumentList `
        $currentTaskDir.FullName, `
        $behaviorDir, `
        $valueDir, `
        $historyLength, `
        $taskId, `
        $pythonExe, `
        $pythonScriptPath, `
        $taskTimeout, `
        $pollInterval

    $tasks += $task
    Write-Output "启动任务${taskId}（Job ID: $($task.Id)），处理目录: $($currentTaskDir.FullName)"
}

# ========== 监控任务（长时间运行+低资源占用）==========
Write-Output ""
Write-Output "========== 监控任务进度（按Ctrl+C可强制终止）=========="
$allCompleted = $false
$totalTimeout = $taskTimeout * 1.2  # 总超时=单任务超时×1.2

while (-not $allCompleted) {
    # 1. 检查总超时
    $elapsedTotalSec = ($(Get-Date) - $globalStartTime).TotalSeconds
    if ($elapsedTotalSec -gt $totalTimeout) {
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 总超时（$($totalTimeout/3600) 小时），强制终止所有未完成任务"
        foreach ($task in $tasks) {
            if ($task.State -eq "Running") {
                Stop-Job -Job $task -Force -ErrorAction SilentlyContinue
                $timeoutCount++
                $failCount++
            }
        }
        break
    }

    # 2. 轮询任务状态（低频率，30秒一次）
    Start-Sleep -Seconds $pollInterval

    # 3. 获取任务状态（兼容PowerShell 2.0）
    $allJobs = Get-Job -Id $tasks.Id -ErrorAction SilentlyContinue
    if (-not $allJobs -or @($allJobs).Count -eq 0) {
        break
    }

    $runningJobs = $allJobs | Where-Object { $_.State -eq "Running" }
    $completedJobs = $allJobs | Where-Object { $_.State -in "Completed", "Failed", "Stopped" }

    # 4. 处理已完成的任务
    foreach ($job in $completedJobs) {
        if ($job.Name -notlike "*_Processed") {
            $job.Name = "$($job.Name)_Processed"
            $taskId = $job.Id - $tasks[0].Id  # 映射回task_0/task_1等

            try {
                $result = Receive-Job -Job $job -ErrorAction Stop
                if ($result -eq $true) {
                    $successCount++
                    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] ✅ 任务$taskId 完成（Job ID: $($job.Id)）"
                } else {
                    $failCount++
                    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] ❌ 任务$taskId 失败（Job ID: $($job.Id)）"
                }
            } catch {
                $failCount++
                $errorMsg = $_.Exception.Message
                Write-Output "[$(Get-Date -Format 'HH:mm:ss')] ❌ 任务$taskId 读取结果出错: $errorMsg"
                # 输出Job详细错误
                if ($job.ChildJobs[0].Error) {
                    $jobErrors = $job.ChildJobs[0].Error | Out-String
                    Write-Output "   详细错误: $jobErrors"
                }
            }
        }
    }

    # 5. 输出进度
    $completedTotal = $successCount + $failCount
    $elapsedHours = [math]::Round($elapsedTotalSec/3600, 2)
    $remainingJobs = @($runningJobs).Count

    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 进度: $completedTotal/$($tasks.Count) 完成 | 成功: $successCount | 失败: $failCount | 超时: $timeoutCount | 剩余: $remainingJobs | 已运行: $elapsedHours 小时"

    # 6. 检查是否全部完成
    if ($remainingJobs -eq 0) {
        $allCompleted = $true
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 所有任务已完成！"
    }
}

# ========== 清理资源+总结 ==========
# 停止所有剩余Job
foreach ($task in $tasks) {
    if ($task.State -eq "Running") {
        Stop-Job -Job $task -Force -ErrorAction SilentlyContinue
    }
    Remove-Job -Job $task -Force -ErrorAction SilentlyContinue
}

# 输出最终统计
Write-Output ""
Write-Output "========== 最终统计 =========="
$totalDurationHours = [math]::Round(($(Get-Date) - $globalStartTime).TotalHours, 2)
Write-Output "总任务数: $($tasks.Count)"
Write-Output "成功: $successCount"
Write-Output "失败: $failCount"
Write-Output "超时终止: $timeoutCount"
Write-Output "总耗时: $totalDurationHours 小时"
Write-Output ""
Write-Output "========== 日志位置 =========="
foreach ($taskDir in $taskDirs) {
    $taskId = $taskDir.Name -replace 'task_', ''
    Write-Output "任务${taskId}调试日志: $(Join-Path -Path $taskDir.FullName -ChildPath "task_${taskId}_debug.log")"
    Write-Output "任务${taskId}错误日志: $(Join-Path -Path $taskDir.FullName -ChildPath "task_${taskId}_error.log")"
}

# 退出码
if ($failCount -eq 0 -and $timeoutCount -eq 0) {
    Write-Output "✅ 所有任务执行成功！"
    exit 0
} else {
    Write-Output "❌ 部分任务失败/超时，请查看日志"
    exit 1
}