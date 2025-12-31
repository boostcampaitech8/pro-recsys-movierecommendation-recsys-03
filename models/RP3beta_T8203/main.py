import os
from config import *
from data import InteractionData
from model import RP3beta
from trainer import RP3Trainer
from make_submission import make_submission


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = InteractionData(BASE_DATA_DIR)
    URM = data.build_urm()

    model = RP3beta(ALPHA, BETA, TOPK)

    trainer = RP3Trainer(
        model,
        data,
        RECALL_K,
        NDCG_K,
        N_NEG
    )
    trainer.offline_validate()

    model.fit(URM)
    make_submission(
        model,
        URM,
        data,
        topk=10,
        save_path=f"{OUTPUT_DIR}/rp3beta_submission.csv"
    )


if __name__ == "__main__":
    main()