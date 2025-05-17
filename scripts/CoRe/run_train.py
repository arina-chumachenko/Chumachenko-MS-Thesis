import os


def main():
    cuda_id = 1
    chp = 300  # checkpoint of the second stage
    exps = {
        ### {concept}: {concept} {placeholder_token} {initializer_token} {superclass} {concept_property}
        "dog6": "dog6 '<dog6>' dog dog live",
        "bear_plushie": "bear_plushie '<bear-plushie>' toy toy object",
    }

    for concept, exp in exps.items():
        stage_name = 'CR_sdxl'

        # ti_emb_attn + db (CoRe)
        exp_num = 1
        with_gram=False
        with_gram_db=0
        add_attn_reg=False
        name = 'ti_emb_attn'
        os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} {name+'_CoRe'} {with_gram} {cuda_id}")
        os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB_CoRe'} {with_gram_db} {add_attn_reg} {cuda_id}")
        os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db_attn_noft
        exp_num = 2
        with_gram=False
        with_gram_db=False 
        add_attn_reg=True
        name = 'ti_emb_attn'
        os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} {name} {with_gram} {cuda_id}")
        os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB_attn_noft'} {with_gram_db} {add_attn_reg} {cuda_id}")
        os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")


if __name__ == '__main__':
    main()
