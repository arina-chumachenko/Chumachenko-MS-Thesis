import os

def main():
    # chp = 1000
    exps = {
        ### {concept}: {placeholder_token} {class} {property}
        "dog6": "sks dog live",
        "bear_plushie": "sks plushie object",
    }

    for concept, exp in exps.items():
        cuda_id=0
        exp_num = 0
        name ='TS'
        gram=0
        v_gram=0
        use_ca=False
        use_sa=False
        use_v_mid=False
        use_v_all=False
        os.system(f"sh train_ts_db_lora_sdxl.sh {concept} {exp} {exp_num} {name} {gram} {v_gram} {use_ca} {use_sa} {use_v_mid} {use_v_all} {cuda_id}")


if __name__ == '__main__':
    main()
