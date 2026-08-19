# Selective Scan 參數化測資產生器

這個工具會依指定參數建立一個隨機 Mamba 模型與隨機 Token IDs，執行一次完整模型，並留下 selective scan 硬體實驗所需的七個檔案。

每次執行都會讀取專案根目錄目前的 `model.py`，在暫存資料夾中完成運算，再把七個指定輸出複製到案例資料夾。因此不會修改 `model.py`、`run_mamba.py`，也不會讓模型的其他除錯檔案散落在專案根目錄。

## 使用方式

在 `Mamba_By_System-Verilog` 資料夾執行：

```powershell
python .\testbench_generator\generate_case.py --L 4 --d-model 16 --N 16 --expand 2 --seed 0
```

輸出預設放在：

```text
testbench_generator/generated_cases/
```

例如上面的命令會建立：

```text
case_L4_D16_N16_E2_B1_seed0/
```

如果同名案例已存在，工具會停止以避免誤蓋資料；確定要重建時加入 `--overwrite`。

## 可調參數

| 參數 | 預設值 | 意義 |
|---|---:|---|
| `--L`、`--seq-len` | 4 | 序列長度 L |
| `--N`、`--d-state` | 16 | 狀態維度 N；N 就是 `d_state` |
| `--d-model` | 16 | 模型輸入與輸出維度 |
| `--expand` | 2 | Mamba 擴張倍率 |
| `--batch-size` | 1 | Batch 數量；目前硬體通常使用 1 |
| `--vocab-size` | 256 | Token 字典大小 |
| `--dt-rank` | auto | delta 投影 rank，可填 `auto` 或正整數 |
| `--d-conv` | 4 | 卷積 kernel 大小 |
| `--seed` | 0 | 隨機種子；相同參數與 seed 可重現相同資料 |
| `--case-name` | 自動命名 | 自訂輸出資料夾名稱 |
| `--output-root` | generated_cases | 自訂案例輸出根目錄 |
| `--overwrite` | 關閉 | 允許覆蓋同名案例的既有輸出 |

`d_inner` 不是獨立輸入，它會由下式產生：

```text
d_inner = d_model * expand
```

目前 `--n-layer` 必須維持 1，因為一個案例對應一個 selective scan 硬體核心；多層模型會重複覆寫同名測試資料。

## 輸出檔案

| 檔案 | 資料數量 | 格式 |
|---|---|---|
| `A_testbench.txt` | `d_inner * N` | 16-bit Hex |
| `B_testbench.txt` | `batch * L * N` | Q8、16-bit Hex |
| `C_testbench.txt` | `batch * L * N` | Q8、16-bit Hex |
| `D_testbench.txt` | `d_inner` | Q8、16-bit Hex |
| `delta_testbench.txt` | `batch * L * d_inner` | Q8、16-bit Hex |
| `u_shape_int.txt` | `batch * L * d_inner` | Q8、16-bit Hex |
| `y_q16_answer.txt` | `batch * L * d_inner` | Q16、32-bit Hex |
| `y_q16_float_answer.txt` | `batch * L * d_inner` | Python 整數近似結果反量化後的 float |
| `y_origin_answer.txt` | `batch * L * d_inner` | 原始 float selective scan 標準答案 |

另外會產生 `experiment_config.json`，記錄本次參數、衍生尺寸、隨機 Token IDs、各檔案行數與格式，方便日後比對不同硬體實驗。

模型權重與 Token IDs 會依 `seed` 隨機產生；A、D 則保留目前 `model.py` 所定義的 Mamba 初始化方式，因此 D 預設仍為 1.0（輸出 Q8 為 `0100`）。

## 其他範例

改變序列長度和狀態維度：

```powershell
python .\testbench_generator\generate_case.py --L 8 --N 32 --d-model 16 --expand 2 --seed 1
```

指定案例名稱：

```powershell
python .\testbench_generator\generate_case.py --L 16 --N 8 --case-name L16_N8_test1
```

## 硬體輸入合併檔

每次產生案例後，工具也會建立 `test_in_0.txt`，內容按照硬體讀取順序直接串接：

```text
A_testbench.txt
B_testbench.txt
delta_testbench.txt
u_shape_int.txt
C_testbench.txt
D_testbench.txt
```

檔案之間不會加入標題或空白行，每一行仍是一筆 16-bit Hex 資料。各區段的開始行、結束行及行數會記錄在 `experiment_config.json` 的 `merged_hardware_input` 欄位中。

## RTL 與標準 float 誤差比較

先把 ModelSim 產生的 `test_out_0.txt` 放進對應的案例資料夾，再執行：

```powershell
python .\testbench_generator\compare_error.py --case-dir .\testbench_generator\generated_cases\case_L4_D16_N16_E2_B1_seed0
```

預設比較對象是：

```text
test_out_0.txt vs y_origin_answer.txt
```

這是硬體 Q16 結果與原始 float 軟體結果的誤差，適合放入不同參數的統計表。程式會計算 MAE、RMSE、中位絕對誤差、P95、最大絕對誤差、相對 MAE、MAE LSB、最大誤差 LSB，以及把 float 四捨五入為 Q16 後的完全相同比例。

輸出包含：

```text
案例資料夾/error_report.json
案例資料夾/error_details.csv
testbench_generator/comparison_summary.csv
```

其中 `comparison_summary.csv` 會集中保留各組 L、d_model、N、expand、seed 與誤差，可直接用 Excel 製作統計表。同一案例與同一參考檔再次比較時會更新原本那一列。

若要確認 RTL 是否重現 Python 的整數近似模型，可改用：

```powershell
python .\testbench_generator\compare_error.py `
  --case-dir .\testbench_generator\generated_cases\case_L4_D16_N16_E2_B1_seed0 `
  --reference .\testbench_generator\generated_cases\case_L4_D16_N16_E2_B1_seed0\y_q16_float_answer.txt
```

注意：`y_q16_float_answer.txt` 不是原始 float 標準答案；它只是 `y_q16_answer.txt / 65536`。真正用來評估硬體量化與 exp 近似誤差的是 `y_origin_answer.txt`。
