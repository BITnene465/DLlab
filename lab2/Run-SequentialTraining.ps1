
# 定义配置文件
$configFiles = @(
    "config1.json",
    "config2.json",
    "config3.json",
    "config4.json"
)

# 训练脚本路径
$trainScript = "train_lstm.py"

Write-Host "开始按顺序执行训练..." -ForegroundColor Green

# 循环执行每个配置文件的训练
foreach ($config in $configFiles) {
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host "开始使用配置文件: $config 进行训练" -ForegroundColor Cyan
    Write-Host "执行命令: python $trainScript --config $config" -ForegroundColor Cyan
    Write-Host "=======================================" -ForegroundColor Cyan
    
    # 执行训练脚本并等待其完成
    python $trainScript --config $config
    
    # 检查上一个命令的执行结果
    if ($LASTEXITCODE -eq 0) {
        Write-Host "使用配置 $config 的训练成功完成" -ForegroundColor Green
    } else {
        Write-Host "使用配置 $config 的训练失败，退出代码: $LASTEXITCODE" -ForegroundColor Red
        # 如果想在失败时继续下一个训练，请注释下一行
        # exit 1
    }
    
    Write-Host ""
}

Write-Host "所有训练任务已完成！" -ForegroundColor Green
