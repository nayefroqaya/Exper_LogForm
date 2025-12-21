from logging import raiseExceptions
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import re
import torch
from sentence_transformers import SentenceTransformer

# =========================
# Preprocessing by block
# =========================
def preprocess_data_by_block(
    df,
    mode,
    log_name,
    block_col="Node_block_id",
    window_size=120
):
    x_data, y_data = [], []

    grouped = df.groupby(block_col, sort=False)

    for block_id, df_blk in tqdm(grouped, desc=f"{mode} blocks"):
        # safety check
        if len(df_blk) != window_size:
            continue

        x_data.append(
            np.array(df_blk["Vector"].tolist())
        )

        labels = df_blk["Label"].tolist()

        # normal window if all logs are normal
        if all(l == "-" for l in labels):
            y = [1, 0]
        else:
            y = [0, 1]

        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    np.savez(
        f"{OUTPUT_DIR}/{log_name}_{mode}_block_w{window_size}.npz",
        x=x_data,
        y=y_data
    )

    print(f"{mode} saved:",
          x_data.shape,
          y_data.shape)


def preprocess_data(df, mode, log_name, window_size=120):
    x_data, y_data = [], []

    if len(df) % window_size != 0:
        raise ValueError(f'Data length must be divisible by {window_size}')

    num_windows = len(df) // window_size

    for i in tqdm(range(num_windows)):
        df_blk = df.iloc[i*window_size:(i+1)*window_size]

        x_data.append(
            np.array(df_blk["Vector"].tolist())
        )

        labels = df_blk["Label"].tolist()

        # normal window if ALL logs are normal
        if all(l == '-' for l in labels):
            y = [1, 0]   # normal
        else:
            y = [0, 1]   # anomaly

        y_data.append(y)

    np.savez(
        f'preprocessed_data/{log_name}_{mode}_w{window_size}_data.npz',
        x=np.array(x_data),
        y=np.array(y_data)
    )



if __name__ == '__main__':
    # =========================
    # Config
    # =========================
    LOG_NAME = "BGL"
    WINDOW_SIZE = 120

    TRAIN_PKL = "data/train.pkl"
    TEST_PKL = "data/test.pkl"

    OUTPUT_DIR = "preprocessed_data"

    # =========================
    # Model
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer("distilbert-base-nli-mean-tokens", device=device)

    # =========================
    # Load data
    # =========================
    df_train = pd.read_pickle(TRAIN_PKL)
    df_test = pd.read_pickle(TEST_PKL)

    # =========================
    # Vector embedding
    # =========================
    print("Vector embedding...")

    # collect unique templates from both splits
    all_templates = pd.concat([df_train["EventTemplate"], df_test["EventTemplate"]]).unique()

    embeddings = model.encode(all_templates, batch_size=128, show_progress_bar=True)

    template_dict = dict(zip(all_templates, embeddings))

    # map vectors
    df_train["Vector"] = df_train["EventTemplate"].map(template_dict)
    df_test["Vector"] = df_test["EventTemplate"].map(template_dict)

    print("Embedding done.")

    # =========================
    # Run preprocessing
    # =========================
    preprocess_data_by_block(df_train, mode="training", log_name=LOG_NAME, window_size=WINDOW_SIZE)

    preprocess_data_by_block(df_test, mode="testing", log_name=LOG_NAME, window_size=WINDOW_SIZE)




    '''
    num_workers = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        'distilbert-base-nli-mean-tokens', device=device)

    # load data
    log_name = 'BGL'
    df_template = pd.read_csv(f"parse_result/{log_name}.log_templates.csv")
    df_structured = pd.read_csv(f"parse_result/{log_name}.log_structured.csv")

    # calculate vectors for all known templates
    print('vector embedding...')
    embeddings = model.encode(
        df_template['EventTemplate'].tolist())  # num_workers=num_workers)
    df_template['Vector'] = list(embeddings)
    template_dict = df_template.set_index('EventTemplate')['Vector'].to_dict()

    # convert templates to vectors for all logs
    vectors = []
    for idx, template in enumerate(df_structured['EventTemplate']):
        try:
            vectors.append(template_dict[template])
        except KeyError:
            # new template
            vectors.append(model.encode(template))
    df_structured['Vector'] = vectors
    print('done')
    df_structured.drop(
        columns=['Date', 'Node', 'Time', 'NodeRepeat', 'Type', 'Component', 'Level'])

    num_windows = len(df_structured)//20
    df_structured = df_structured.iloc[:num_windows*20]

    training_windows = (num_windows//5)*4
    df_structured['Usage'] = 'testing'
    df_structured.iloc[:training_windows*20,
                       df_structured.columns.get_loc('Usage')] = 'training'

    df_test = df_structured[df_structured['Usage'] == 'testing']
    df_train = df_structured[df_structured['Usage'] == 'training']

    # preprocess data
    preprocess_data(df_train, 'training')
    preprocess_data(df_test, 'testing')
    '''
