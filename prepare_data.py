import pandas as pd
from datasets import load_dataset
import os

if not os.path.exists("data"):
    os.makedirs("data")
print("网络环境已就绪，正在从 Hugging Face 加载金融 SFT 数据集...")

try:
    dataset = load_dataset('gbharti/finance-alpaca', split='train')

    df=pd.DataFrame(dataset)

    df_small = df[['instruction', 'output']].head(1000)

    output_path="data/finance_data.jsonl"
    df_small.to_json(output_path,orient="records",lines=True,force_ascii=False)
    print(f"✅ 成功！已从 Hugging Face 获取 {len(df_small)} 条真实金融数据。")
    print(f"文件位置: {output_path}")
    print(f"数据示例: {df_small.iloc[0]['instruction'][:50]}...")

except Exception as e:
    print(f"❌ 下载依然遇到障碍: {e}")