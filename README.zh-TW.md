# ftlsim — SSD FTL 模擬器

[English](README.md) | **繁體中文**

從零用 C++17 寫的 SSD Flash Translation Layer 模擬器，以 I/O trace 驅動。
用來量測 GC 策略、預留空間（OP）、資料擺放（冷熱分離）和 wear leveling 對寫入放大（WAF）的影響。
最終目標是比較規則式擺放（SepBIT）和韌體放得下的 AI 資料壽命預測器。
完整計畫見 [01-ftl-simulator-plan.md](01-ftl-simulator-plan.md)。

## 功能

| 項目 | 已實作 |
|---|---|
| NAND 模型 | 以 page 寫入、以 block 抹除、不能原地覆寫；記錄每個 block 的抹除次數與 OOB 中的 LPN |
| 對應表 | page-level L2P |
| GC 受害 block 挑選 | `greedy`、`cost-benefit`、`fifo` |
| 資料擺放 | `none`、`sepgc`（GC 搬移的資料另開寫入點）、`wcount`（依寫入次數）、`sepbit`（FAST'22）、`oracle`（知道真實壽命）、`fdp`（主機提示，可加雜訊或只給粗分類，GC 可選 persistent／initial isolation）、`ml`（決策樹壽命預測器） |
| Wear leveling | 動態（`--wear-aware`：配置抹除次數最少的空 block）；靜態（`--static-wl=N`：抹除次數差距超過 N 時，把冷資料搬到抹除次數最高的空 block） |
| 學習式擺放 | 線上特徵，每個 LPN 只需 2 bytes 狀態；決策樹離線訓練（`scripts/train_predictor.py`），載入後約 1 KB |
| 主機操作 | 寫入、讀取、TRIM |
| Workload | `seq`、`uniform`、`hotcold`（80/20）、`zipf`、真實 trace（MSR／Alibaba／Tencent） |
| 正確性 | 每次讀取都用影子對照表檢查，外加 L2P／OOB 完整一致性檢查 |

## 編譯與測試

需要 `g++`（C++17）與 `make`，開發環境為 WSL2 Ubuntu。

```bash
make            # 產生 build/ftlsim 和 build/test_ftl
make test       # 單元測試與理論驗證
bash scripts/smoke_trace.sh   # 用 --verify 跑一次完整的 trace 流程
```

## 使用方式

```bash
./build/ftlsim --workload=zipf --gc=cost-benefit --placement=sepbit --op=0.14
./build/ftlsim --help
```

輸出一個 JSON 物件：WAF、GC 次數、每個受害 block 平均搬幾個有效 page、抹除次數的最小／最大／平均／標準差，以及 FIFO 解析模型的 WAF 供對照。

執行順序：先循序寫滿整顆裝置 → `--warmup` 倍容量的寫入（不計入量測）→ `--writes` 倍容量的寫入（量測）。整段執行會先錄下來再重播，所以每種擺放方式拿到的 I/O 序列完全相同，`oracle`／`fdp` 也能查到每個 page 的真實壽命。

### 學習式擺放

```bash
bash scripts/ml_pipeline.sh zipf 0.14     # 匯出特徵（seed 1）→ 訓練 → 在 seed 2 上評估
./build/ftlsim --placement=ml --model=results/ml/zipf.tree --ml-gc=sepbit
```

訓練需要 [uv](https://docs.astral.sh/uv/)，相依套件直接寫在腳本裡。

### 真實 trace

```bash
python3 tools/convert_trace.py alibaba io_traces.csv --volume 28 -o vol28.csv
./build/ftlsim --trace=vol28.csv --size-to-trace --trace-loops=2 --trace-warmup-loops=1
```

統一格式：`timestamp_us,op,offset_bytes,size_bytes`，`op` 為 R、W、T 其中之一。

### 實驗矩陣

```bash
python3 scripts/experiment.py --grid op_sweep  --out results/op_sweep.csv
python3 scripts/experiment.py --grid placement --out results/placement.csv
python3 scripts/summarize.py results/placement.csv --rows workload,gc,op --cols placement
```

## 驗證

| 檢查 | 結果 |
|---|---|
| 循序覆寫，所有 GC 策略 | WAF = 1.00000，GC 搬移 0 個 page |
| 影子對照表：3 種 GC × 4 種擺放 × wear leveling 開關，Zipf 寫入 | 每次讀到的都是最後一次寫入的版本 |
| 均勻隨機寫入下 FIFO 與 Desnoyers（SYSTOR'12）解析公式比對 | OP 7%／14%／28% 誤差分別為 0.7%／0.3%／0.1% |
| 均勻隨機寫入下 greedy ≤ FIFO | 成立 |
| 錄製重播 vs. 直接執行 | NAND 寫入次數完全相同 |
| 靜態 WL、oracle、FDP、ML 擺放的影子對照表檢查 | 每次讀到的都是最後一次寫入的版本 |

解析模型是和「有效 OP」比對：扣掉 GC 保留的 free block 門檻，以及 open block 還沒寫滿的部分。

## 初步結果

2048 個 block × 每 block 256 page × 4 KiB（實體 2 GiB），預熱 3 倍容量、量測 5 倍容量。

**OP 與 GC 策略對 WAF 的影響（不分冷熱）**

| Workload | GC | 7% | 10% | 14% | 20% | 28% |
|---|---|---|---|---|---|---|
| uniform | greedy | 7.73 | 5.63 | 4.23 | 3.17 | 2.47 |
| uniform | cost-benefit | 9.17 | 6.43 | 4.68 | 3.39 | 2.58 |
| zipf 0.99 | greedy | 7.50 | 5.85 | 4.62 | 3.59 | 2.84 |
| zipf 0.99 | cost-benefit | 5.79 | 4.36 | 3.42 | 2.74 | 2.29 |

**各種擺放方式的 WAF（greedy GC，OP 14%）**

| Workload | none | sepgc | wcount | sepbit |
|---|---|---|---|---|
| uniform | 4.23 | 4.24 | 4.25 | 4.33 |
| hotcold 80/20 | 4.53 | 4.53 | 4.46 | 4.43 |
| zipf 0.99 | 4.62 | 4.59 | 4.22 | 3.55 |

目前觀察：cost-benefit 只有在 workload 偏斜時才有優勢；均勻隨機寫入下分冷熱沒有幫助，多開的寫入點反而吃掉 OP；SepBIT 在 Zipf、OP 14% 下讓 WAF 降低 23%。

**Wear leveling（80% 冷資料永遠不再寫入，OP 7%，greedy）**

| 設定 | WAF | 最大抹除次數 | 抹除次數標準差 |
|---|---|---|---|
| 關閉 | 2.15 | 106 | 44.7 |
| 靜態，差距 ≤ 20 | 2.30 | 53 | 4.9 |
| 靜態，差距 ≤ 50 | 2.24 | 80 | 19.2 |

靜態 WL 在這裡多付出 7% 的 WAF。冷資料搬去哪裡很關鍵：在單元測試中，如果寫回和主機共用的寫入點，WAF 會從 1.16 升到 3.68；改成放到抹除次數最高的空 block，只升到 1.33。

**壽命預測 vs. 上下限（Zipf 0.99，OP 14%，seed 1 訓練 → seed 2 測試）**

| 擺放方式 | WAF |
|---|---|
| none | 4.62 |
| 只靠 GC 依年齡重新分組（FDP initial，隨機提示） | 4.07 |
| ml，深度 6 的決策樹，約 1.2 KB | 3.78 – 3.97 |
| sepbit | 3.55 |
| FDP，完美主機提示，persistent isolation | 1.93 |
| 完美主機分類 + GC 依年齡重新分組 | 1.74 |
| oracle | 1.56 |

在無記憶性的合成 workload 上，決策樹贏不了 SepBIT：能學到的只有每個 page 的寫入頻率，而 SepBIT 用「上一次壽命」搭配自動調整的門檻已經抓到大部分。oracle 的 1.56 需要知道隨機的覆寫間隔，任何根據歷史的預測器都看不到，所以在這裡只是很寬鬆的上限。真正的考驗是有時間與空間模式的真實 trace。

**FDP 主機提示（同一組設定）**

| 提示 | Persistent isolation（GC 留在原 RUH） | Initial isolation（SSD 依年齡重新分組 GC 資料） |
|---|---|---|
| 完美 | 1.93 | 1.74 |
| 只分冷熱兩類 | 3.83 | – |
| 20% 錯誤 | 4.42 | 4.02 |
| 50% 錯誤 | 5.03（比不分冷熱的 4.62 還差） | 4.14 |

在 persistent isolation 下，錯誤的提示比沒有提示還糟；讓 SSD 自己重新分組 GC 資料，可以救回大部分損失。

## 目錄結構

```
src/
  nand.{h,cpp}        NAND 陣列：page、block、抹除次數、OOB
  ftl.{h,cpp}         page-mapped FTL：寫入／讀取／TRIM、GC、free block 池、一致性檢查
  gc_policy.{h,cpp}   受害 block 挑選：greedy／cost-benefit／fifo
  placement.{h,cpp}   寫入點分配：none／sepgc／wcount／sepbit／oracle／fdp
  predictor.{h,cpp}   線上特徵、決策樹、ml 擺放
  workload.{h,cpp}    合成 workload 產生器與統一格式 trace 讀取
  sim.{h,cpp}         錄製／重播、資料死亡時間、特徵匯出、抹除次數統計、FIFO 解析模型
  main.cpp            命令列介面、JSON 輸出
tests/test_ftl.cpp    正確性與理論驗證測試
tools/convert_trace.py  MSR／Alibaba／Tencent 轉成統一格式
scripts/              實驗矩陣、結果彙整、smoke test
```

## 進度

- [x] 第 1 週：NAND 模型、L2P、影子對照表、循序寫入 WAF = 1
- [x] 第 2 週：greedy／cost-benefit／FIFO GC、OP 掃描、解析模型驗證、SepBIT、GC 資料分流
- [x] 第 3 週（程式）：trace 轉換工具、靜態 wear leveling
- [ ] 第 3 週（資料）：跑 MSR／Alibaba／Tencent trace，和 SepBIT 論文比對趨勢
- [x] 第 4 週（合成 workload）：oracle、FDP 提示（雜訊／粗分類、persistent／initial）、ML 預測器流程
- [ ] 第 4 週（真實 trace）：volume A 訓練、volume B 測試；時段特徵；線上重新訓練
- [ ] 第 5 週：圖表、技術文章
