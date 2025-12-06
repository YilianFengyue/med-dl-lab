# ============================================
# run.ps1 - PowerShell 运行脚本
# ============================================

# 激活虚拟环境
Write-Host "🔧 激活虚拟环境..." -ForegroundColor Cyan
& "D:/code/Python/.venv/Scripts/Activate.ps1"

# 检查是否成功
if (-not $?) {
    Write-Host "❌ 虚拟环境激活失败!" -ForegroundColor Red
    exit 1
}

# 切换到代码目录
Set-Location "D:\code\Python\Exp6\src\part1_pytorch"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  肺炎图像识别系统 - 运行选项" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "1. 完整流程 (训练 + 评估)"
Write-Host "2. 只训练"
Write-Host "3. 只评估"
Write-Host "4. 查看模型信息"
Write-Host "5. 快速测试 (5 epochs)"
Write-Host ""

$choice = Read-Host "请选择 [1-5]"

switch ($choice) {
    "1" { python main.py }
    "2" { python main.py --train }
    "3" { python main.py --eval }
    "4" { python main.py --info }
    "5" { python main.py --ae-epochs 5 --cnn-epochs 5 }
    default { 
        Write-Host "无效选择，运行完整流程..." -ForegroundColor Yellow
        python main.py 
    }
}

Write-Host ""
Write-Host "✅ 完成!" -ForegroundColor Green
Read-Host "按回车键退出"


# ============================================
# 直接命令行运行方式 (复制到终端)
# ============================================
<#
# 完整流程
cd D:\code\Python\Exp6\src\part1_pytorch
python main.py

# 只训练
python main.py --train

# 只评估  
python main.py --eval

# 查看模型信息
python main.py --info

# 自定义参数
python main.py --ae-epochs 100 --cnn-epochs 100 --lr 0.0005 --batch-size 16

# 快速测试 (5 epochs)
python main.py --ae-epochs 5 --cnn-epochs 5
#>