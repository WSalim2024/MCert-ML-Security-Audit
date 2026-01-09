import pandas as pd
import numpy as np


def create_dummy_csv():
    np.random.seed(42)
    print("🎲 Generating multi-class data...")

    # Generate 100 rows, 4 feature columns
    data = np.random.rand(100, 4)

    df = pd.DataFrame(data, columns=['Feat1', 'Feat2', 'Feat3', 'Feat4'])

    # FIX: Generate only 3 classes (0, 1, 2) to ensure adequate samples per class
    # With 100 samples and 3 classes, each class gets ~33 samples on average
    df['Target'] = np.random.randint(0, 3, size=100)

    # Safety guarantee: Ensure at least 2 samples per class for stratification
    # Force first 6 rows to have 2 of each class
    df.loc[0, 'Target'] = 0
    df.loc[1, 'Target'] = 0
    df.loc[2, 'Target'] = 1
    df.loc[3, 'Target'] = 1
    df.loc[4, 'Target'] = 2
    df.loc[5, 'Target'] = 2

    df.to_csv('user_data.csv', index=False)

    print(f"✅ Created 'user_data.csv' (Rows: {len(df)})")
    print(f"   Target Distribution: {df['Target'].value_counts().sort_index().to_dict()}")
    print(f"   Min samples per class: {df['Target'].value_counts().min()}")


if __name__ == "__main__":
    create_dummy_csv()
