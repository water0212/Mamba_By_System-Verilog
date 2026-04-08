import torch
from model import Mamba, ModelArgs

# 1. 設定迷你模型的參數 (為了硬體模擬，我們把維度設小一點)
args = ModelArgs(
    d_model=16,      # 隱藏層維度 (FPGA 測試先用小的)
    n_layer=1,       # 只用 1 層 Mamba Block
    vocab_size=256,  # 字典大小
    d_state=16,      # 狀態維度 (N)
    expand=2         # 擴展係數
)

# 2. 初始化模型
model = Mamba(args)
model.eval() # 設定為推論模式

# 3. 準備測試輸入資料
batch_size = 1
seq_len = 4  # 序列長度 (測試先用 4 個 Token)
# 隨機產生輸入的 Token ID
input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))

print(f"輸入的 Token IDs: \n{input_ids}\n")

# 4. 執行前向傳播 (Forward Pass)
with torch.no_grad(): # 推論時不需要計算梯度
    logits = model(input_ids)

print(f"輸出的 Logits 形狀: {logits.shape}")
print(f"輸出數值 (部分): \n{logits[0, 0, :5]}")
