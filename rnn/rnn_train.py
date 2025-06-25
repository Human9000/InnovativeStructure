# train_predict.py

# --- Standard Imports ---
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
# --- Import the user's model ---
try:
    from sort_gru_v1 import SortGRU
except ImportError:
    print("ERROR: Could not import SortGRU from my_model.py.")
    print("Please ensure 'my_model.py' contains your model definition and is in the same directory or accessible via PYTHONPATH.")
    exit()

# --- Data Handling ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data_features, data_target, input_seq_len, output_seq_len):
        self.data_features = data_features
        self.data_target = data_target
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        return len(self.data_features) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        # Input sequence for the model
        x = self.data_features[idx : idx + self.input_seq_len + self.output_seq_len]
        # Target sequence (the actual future values we want to predict)
        # y = self.data_target[idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len]
        return torch.FloatTensor(x)#, torch.FloatTensor(y)


def create_dummy_csv(filename="dummy_data.csv", rows=1000, period=24*3):
    print(f"Creating dummy CSV: {filename} with {rows} rows and period {period}.")
    time_index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows, freq='H'))
    t = np.arange(rows)

    data = {
        'Time': time_index,
        # 特征1: 强周期性 + 线性趋势 + 噪声
        'F1': (np.sin(2 * np.pi * t / period) * 10 +  # 周期部分
               np.sin(2 * np.pi * t / (period / 2)) * 3 + # 次级周期
               t * 0.01 +                               # 线性趋势
               np.random.randn(rows) * 1.5),            # 噪声

        # 特征2: 周期性（余弦）+ 不同的相位和幅度 + 噪声
        'F2': (np.cos(2 * np.pi * t / period + np.pi/4) * 8 + # 周期部分（相位偏移）
               np.random.randn(rows) * 1.0),                  # 噪声

        # 特征3: 周期性 + 更高频率的周期叠加 + 噪声
        'F3': (np.sin(2 * np.pi * t / period) * 5 +
               np.cos(2 * np.pi * t / (period * 0.75)) * 2.5 + # 非整数倍周期
               np.random.randn(rows) * 0.8),

        # 特征4: 线性趋势 + 较弱的周期性 + 噪声
        'F4': (t * 0.05 +                                  # 较强线性趋势
               np.sin(2 * np.pi * t / (period * 1.5)) * 2 + # 较弱、较长周期
               np.random.rand(rows) * 3 - 1.5),             # 均匀分布噪声

        # 特征5: 模拟一个受周期性影响的“事件”或“状态”特征 (0或1附近的值，受周期影响)
        'F5': (0.5 + 0.4 * np.sin(2 * np.pi * t / period + np.pi/2) + # 基础周期性概率
               0.2 * np.cos(2 * np.pi * t / (period / 3)) +   # 更高频影响
               (np.random.rand(rows)-0.5)*0.3) # 噪声
    }
    # 对F5进行裁剪，使其更像一个概率或受限值
    data['F5'] = np.clip(data['F5'] * 10, 0, 12) # 放大并裁剪到一个范围，例如0-12

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dummy CSV '{filename}' created with period {period}.")
    
    # 可选：绘制生成的数据以供检查
    if rows <= 5000: # 避免绘制过大的数据集
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(['F1', 'F2', 'F3', 'F4', 'F5']):
            plt.subplot(5, 1, i+1)
            plt.plot(df['Time'][:min(rows, period * 4)], df[col][:min(rows, period*4)]) # 绘制前几个周期
            plt.title(col)
        plt.tight_layout()
        plt.savefig(f"{filename.split('.')[0]}_generated_data_preview.png")
        print(f"Generated data preview saved to {filename.split('.')[0]}_generated_data_preview.png")
        plt.show()

    return filename


if __name__ == '__main__':
    # --- Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    WINDOW_SIZE = 24          # Patching window for SortGRU internal ops (from your model usage)
    INPUT_SEQ_STEPS = 24 * 4  # 96 historical time steps for input
    TARGET_SEQ_STEPS = 24     # 24 future time steps to predict

    NUM_FEATURES = 5          # F1, F2, F3, F4, F5 (as per your CSV description)
    ENCODER_CHANNELS = 64     # Hidden dim for encoder, input to decoders.
                              # Make sure this is compatible with your model's group settings.
    SEQ_OUTPUT_CHANNELS = NUM_FEATURES # Predicting all 5 features in sequence
    ONE_OUTPUT_CHANNELS = 0   # Not using one_decoder for this task, so 0 output channels
    DECODER_GROUPS = 8        # Groups for decoders. ENCODER_CHANNELS must be divisible by this.
    NUM_GRU_LAYERS = 2        # Number of layers in SortGRUEncoder

    BATCH_SIZE = 64
    LEARNING_RATE = 0.003
    EPOCHS = 10 # Adjust as needed

    # --- Crucial Divisibility Checks (based on the imported model's structure) ---
    # These checks are based on the internal logic of your provided model classes.
    # For SortAttn (default groups=4 in SortGRUCell):
    # ENCODER_CHANNELS * 2 must be divisible by SortAttn's `groups` (default 4).
    sort_attn_groups = 4 # Default in your SortAttn within SortGRUCell
    if (ENCODER_CHANNELS * 2) % sort_attn_groups != 0:
        print(f"CRITICAL WARNING: For SortAttn, (ENCODER_CHANNELS * 2) = {ENCODER_CHANNELS*2} "
              f"is not divisible by SortAttn's groups = {sort_attn_groups}. "
              f"This will likely cause a runtime error or incorrect behavior in `my_model.SortAttn`.")

    # For SeqDecoder and OneEncoder (if used):
    # ENCODER_CHANNELS must be divisible by DECODER_GROUPS.
    if SEQ_OUTPUT_CHANNELS > 0 and ENCODER_CHANNELS % DECODER_GROUPS != 0:
        print(f"CRITICAL WARNING: ENCODER_CHANNELS ({ENCODER_CHANNELS}) "
              f"is not divisible by DECODER_GROUPS ({DECODER_GROUPS}). "
              f"This will likely cause a runtime error or incorrect behavior in `my_model.SeqDecoder`.")
    if ONE_OUTPUT_CHANNELS > 0 and ENCODER_CHANNELS % DECODER_GROUPS != 0:
         print(f"CRITICAL WARNING: ENCODER_CHANNELS ({ENCODER_CHANNELS}) "
              f"is not divisible by DECODER_GROUPS ({DECODER_GROUPS}). "
              f"This will likely cause a runtime error or incorrect behavior in `my_model.OneEncoder`.")


    # --- 1. Load and Prepare Data ---
    # Create dummy data if your CSV doesn't exist. Replace with your actual CSV path.
    # The CSV should have columns: Time, F1, F2, F3, F4, F5
    csv_file = "my_timeseries_data.csv"
    try:
        df = pd.read_csv(csv_file)
        if not all(col in df.columns for col in ['F1', 'F2', 'F3', 'F4', 'F5']):
            print(f"CSV {csv_file} does not contain all required feature columns. Creating dummy data.")
            create_dummy_csv(csv_file, rows=INPUT_SEQ_STEPS + TARGET_SEQ_STEPS + 6500)
            df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"CSV file '{csv_file}' not found. Creating dummy data.")
        create_dummy_csv(csv_file, rows=INPUT_SEQ_STEPS + TARGET_SEQ_STEPS + 6500)
        df = pd.read_csv(csv_file)


    feature_cols = ['F1', 'F2', 'F3', 'F4', 'F5']
    features_np = df[feature_cols].values.astype(np.float32)

    # --- 2. Standardize Features ---
    # Split data notionally for fitting scaler (in real scenario, use a dedicated train set)
    split_idx = int(len(features_np) * 0.8)
    if split_idx == 0 and len(features_np) > 0 : split_idx = len(features_np) # handle very small datasets for dummy case
    
    train_features_for_scaling = features_np[:split_idx]
    
    scaler = StandardScaler()
    if len(train_features_for_scaling) > 0:
        scaler.fit(train_features_for_scaling)
        scaled_features = scaler.transform(features_np)
    else: # Not enough data to fit scaler, use unscaled data (not recommended for real use)
        print("Warning: Not enough data to fit scaler. Using unscaled data. This may affect performance.")
        scaled_features = features_np


    # We use the same scaled_features for input (X) and for deriving targets (Y)
    train_data_len = split_idx # Use the "training" portion for creating dataset
    
    # Ensure there's enough data for at least one sample
    min_data_needed = INPUT_SEQ_STEPS + TARGET_SEQ_STEPS
    if train_data_len < min_data_needed:
         print(f"Warning: Training data length ({train_data_len}) is less than minimum needed ({min_data_needed}). "
               "Attempting to use all available data for training dataset.")
         dataset_features = scaled_features
    else:
         dataset_features = scaled_features[:train_data_len]

    train_dataset = TimeSeriesDataset(
        dataset_features, # Input features source
        dataset_features, # Target features source (shifted in time by TimeSeriesDataset)
        INPUT_SEQ_STEPS,
        TARGET_SEQ_STEPS
    )
    
    if len(train_dataset) == 0:
        print(f"ERROR: Not enough data to create any training samples. "
              f"Need at least {min_data_needed} rows in the (scaled) feature data used for the dataset. "
              f"Available for dataset: {len(dataset_features)} rows.")
        exit()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")


    # --- 3. Initialize Model, Loss, Optimizer ---
    # Instantiating the imported model
    model = SortGRU(
        window=WINDOW_SIZE,
        in_ch=NUM_FEATURES,
        enc_ch=ENCODER_CHANNELS,
        out_ch_seq=SEQ_OUTPUT_CHANNELS,
        out_ch_one=ONE_OUTPUT_CHANNELS, # Set to 0 as we only use seq_decoder
        dec_groups=DECODER_GROUPS,
        n_layers=NUM_GRU_LAYERS
    ).to(DEVICE)

    model.use_mem(False) # IMPORTANT: For batch training with shuffle, reset memory between batches.

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel Architecture (Imported from my_model.py):")
    # print(model) # You can uncomment this to see the structure
 
    # --- 4. Training Loop ---
    print("\n--- Starting Training ---") 
    epoch_losses = [] # <--- 新增: 用于存储每个 epoch 的平均损失
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x_batch) in enumerate(train_dataloader):
            x_batch = x_batch.to(DEVICE)
            # print(x_batch.shape, TARGET_SEQ_STEPS)
            optimizer.zero_grad()
            _, y_seq_pred_full, _ = model(x_batch[:,:-TARGET_SEQ_STEPS])  
            loss = (y_seq_pred_full - x_batch[:,TARGET_SEQ_STEPS:]).pow(4).mean() * 100
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % max(1, len(train_dataloader) // 5) == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.6f}")

                if epoch > 18:
                    # 绘制当前轮的x_batch和y_batch_true_future, y_seq_pred_full
                    plt.figure(figsize=(12, 8))
                    for i in  range(5):
                        plt.subplot(5, 1, 1 + i)
                        plt.plot(np.linspace(0, 96+24, 96+24), x_batch[0, :, i].detach().cpu().numpy(), label='x_batch') 
                        plt.plot(np.linspace(24, 96 +24, 96),  y_seq_pred_full[0, :, i].detach().cpu().numpy(), label='y_seq_pred_full') 
                        plt.legend()  

                    plt.show()



        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss) # <--- 新增: 记录当前 epoch 的平均损失
        print(f"Epoch {epoch+1}/{EPOCHS} finished. Average Training Loss: {avg_loss:.6f}")

    print("--- Training finished. ---")

    # --- 可视化训练损失 ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', linestyle='-')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig("training_loss_plot.png") # 保存图像
    print("Training loss plot saved to training_loss_plot.png")
    # plt.show() # 如果你想在脚本运行时直接显示图像，取消这行注释


    # --- 5. Inference Example ---
    print("\n--- Starting Inference Example ---")
    model.eval()
    model.use_mem(True)
    model._reset_mem()

    predicted_sequence_original_scale = None # 初始化
    actual_future_original_scale = None    # 初始化
    
    l = len(scaled_features) // TARGET_SEQ_STEPS * TARGET_SEQ_STEPS
    scaled_features = scaled_features[:l]
    if len(scaled_features) < INPUT_SEQ_STEPS:
         print("Not enough data in 'scaled_features' for an inference sample.")
    else:
        inference_input_scaled = scaled_features[:-TARGET_SEQ_STEPS]
        inference_input_tensor = torch.FloatTensor(inference_input_scaled).unsqueeze(0).to(DEVICE)
        print(TARGET_SEQ_STEPS)
        print(scaled_features.shape)
        print(inference_input_tensor.shape)
        print(f"\nInference input shape: {inference_input_tensor.shape}")
        with torch.no_grad():
            _, y_seq_pred_full_inf, _ = model(inference_input_tensor)
 
        pred_seq = y_seq_pred_full_inf[0].cpu().numpy()
        fig, axes = plt.subplots(5, 1, figsize=(60, 5 ), sharex=True)  
        time_steps = np.arange(y_seq_pred_full_inf.shape[1])

        for i in range(5):
            axes[i].plot(time_steps, inference_input_scaled[:, i], label=f'Actual F{i+1}')
            axes[i].plot(time_steps+TARGET_SEQ_STEPS, pred_seq[:, i], label=f'Predicted F{i+1}', linestyle='--')
            axes[i].set_ylabel(f'Feature F{i+1} Value')
            axes[i].legend()
            axes[i].grid(True)

        # axes[-1].set_xlabel(f'Time Steps into the Future (next {TARGET_SEQ_STEPS} steps)')
        fig.suptitle('Inference: Actual vs. Predicted Features', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应标题
        plt.savefig("inference_results_plot.png") # 保存图像
        print("Inference results plot saved to inference_results_plot.png")
        # plt.show() # 如果你想在脚本运行时直接显示图像，取消这行注释
        

    model._reset_mem()
    print("--- Inference Example Finished. ---")