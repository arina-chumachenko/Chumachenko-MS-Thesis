
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

def main():
    cuda_id = 1
    chp = 300  # checkpoint of the second stage
    exps = {
        ### {concept}: {concept} {placeholder_token} {initializer_token} {superclass} {concept_property}
        "cat": "cat '<cat>' cat cat live",
        "dog6": "dog6 '<dog6>' dog dog live",
        "teapot": "teapot '<teapot>' teapot teapot object",
        "bear_plushie": "bear_plushie '<bear-plushie>' toy toy object",
        "colorful_sneaker": "colorful_sneaker '<sneaker>' sneaker sneaker object",
        
        # "grey_sloth_plushie": "grey_sloth_plushie '<sloth-plushie>' toy toy object",
        # "shiny_sneaker": "shiny_sneaker '<sneaker>' sneaker sneaker object",

        # "cat2": "cat2  '<cat2>' cat cat live",
        # "duck_toy": "duck_toy '<ducktoy>' toy toy object",
        # "teapot": "teapot '<teapot>' teapot teapot object",

        # "clock": "clock '<clock>' clock clock object",
        # "backpack": "backpack '<backpack>' backpack backpack object",
        # "berry_bowl": "berry_bowl '<bowl>' bowl bowl object",
        # "red_cartoon": "red_cartoon '<cartoon>' cartoon cartoon object",
        # "poop_emoji": "poop_emoji '<emoji>' emoji toy object",

        # "dog": "dog '<dog>' dog dog live",
        # "monster_toy": "monster_toy '<toy>' toy toy object",
        # "grey_sloth_plushie": "grey_sloth_plushie '<sloth-plushie>' toy toy object",
        
        # "vase": "vase '<vase>' vase vase object",
        # "dog6": "dog6 '<dog6>' dog dog live",
        
        # # "dog5": "dog5 '<dog5>' dog dog live",

        # # "dog2": "dog2 '<dog2>' dog dog live",
        # "can": "can '<can>' can can object",
        # "shiny_sneaker": "shiny_sneaker '<sneaker>' sneaker sneaker object",
        # # "fancy_boot": "fancy_boot '<boot>' boot boot object",
        # "pink_sunglasses": "pink_sunglasses '<sunglasses>' glasses glasses object",

        # # "rc_car": "rc_car '<car>' toy toy object",
        # "dog7": "dog7 '<dog7>' dog dog live",
        # "dog8": "dog8 '<dog8>' dog dog live",
        # "backpack_dog": "backpack_dog '<backpack>' backpack backpack object",
        # "cat_toy": "cat_toy '<toy>' toy toy object",
        
        # "elephant": "elephant '<elephant>' toy toy object",
        # "mug_skulls": "mug_skulls '<skulls>' toy toy object",
    }

    for concept, exp in exps.items():
        stage_name = 'CR_sdxl'

        # ti_emb_attn + db (CoRe)
        exp_num = 0
        with_gram=False
        with_gram_db=0
        add_attn_reg=False
        name = 'ti_emb_attn'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} {name+'_CoRe'} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB_CoRe'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db
        exp_num = 1
        with_gram=False
        add_attn_reg=False
        name = 'ti'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 0 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 2
        with_gram=False
        add_attn_reg=False
        name = 'ti_emb'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_attn + db
        exp_num = 3
        with_gram=False
        add_attn_reg=False
        name = 'ti_attn'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_gram + db
        exp_num = 4
        with_gram=True
        add_attn_reg=False
        name = 'ti_emb_gram'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db_attn_noft
        exp_num = 5
        with_gram=False
        with_gram_db=False 
        add_attn_reg=True
        name = 'ti_emb_attn'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB_attn_noft'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db_gram_noft
        exp_num = 6
        with_gram=False
        with_gram_db=True
        add_attn_reg=True
        name = 'ti_emb_attn'
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} {name+'+DB_gram_noft'} {with_gram_db} {add_attn_reg} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti
        exp_num = 7
        with_gram=0
        name = 'ti'
        os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 0 {exp_num} {name} {with_gram} {cuda_id}")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")



        stage_name = 'CR_comb_sdxl'
        # ti_sdxl_emb_attn + db
        exp_num = 0
        lambda_attn = 0.05
        lambda_emb = 1.5e-4
        # os.system(f"sh train_cr_comb_sdxl.sh {exp} 1 1 {exp_num} 'CR_comb_sdxl_emb_attn' {lambda_emb} {lambda_attn}")


        stage_name ='CR_sdxl'

        # ti_sdxl_emb_attn + db 
        exp_num = 0
        lambda_attn = 0.05
        lambda_emb = 1.5e-4
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} 'ti_sdxl_emb_attn (orig CoRe)' {lambda_emb} {lambda_attn} -C 'type_e'")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} 'ti_sdxl_emb_attn_DB (orig CoRe)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 100")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 200")

        # ti_sdxl + db
        exp_num = 1
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 0 {exp_num} 'ti_sdxl' {lambda_emb} {lambda_attn} -C 'type_e'")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} 'ti_sdxl_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 200")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 300")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 400")
        
        # ti_sdxl_emb_attn_opt + db
        exp_num = 17
        lambda_emb=0.01 
        lambda_attn=0.5
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 1 {exp_num} 'ti_sdxl_emb_attn_opt' {lambda_emb} {lambda_attn} -C 'type_e'")
        # os.system(f"sh train_db_sdxl_with_regs.sh {exp} {exp_num} 'ti_sdxl_emb_attn_opt_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 200")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 300")
        # os.system(f"sh inference_sdxl.sh 00{exp_num}-res-{concept}_{stage_name} 400")
        

        # ti_sdxl_emb + db 
        exp_num = 2
        lambda_emb = 1.5e-4
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_attn + db 
        exp_num = 3
        lambda_attn = 0.05
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} 'ti_sdxl_attn_{lambda_attn}' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_attn_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 4
        lambda_emb = 1e-8
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 5
        lambda_emb = 1e-6
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 6
        lambda_emb = 1e-5
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 7
        lambda_emb = 1e-3
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 8
        lambda_emb = 1e-2
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 9
        lambda_emb = 1
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 10
        lambda_emb = 1e2
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 11
        lambda_emb = 1e8
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 1 0 {exp_num} 'ti_sdxl_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_emb + db 
        exp_num = 12
        lambda_attn = 0.0005
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} 'ti_sdxl_attn_{lambda_attn}' {lambda_attn} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_sdxl_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_sdxl_attn + db 
        exp_num = 13
        lambda_attn = 0.005
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} 'ti_sdxl_attn_{lambda_attn}' {lambda_attn} -C 'type_e'")
        
        # ti_sdxl_attn + db 
        exp_num = 14
        lambda_attn = 0.5
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} 'ti_sdxl_attn_{lambda_attn}' {lambda_attn} -C 'type_e'")
        
        # ti_sdxl_attn + db 
        exp_num = 15
        lambda_attn = 5
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} 'ti_sdxl_attn_{lambda_attn}' {lambda_attn} -C 'type_e'")

        # ti_sdxl_attn + db 
        exp_num = 16
        lambda_attn = 50 
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 1 {exp_num} 'ti_sdxl_attn_{lambda_attn}' {lambda_attn} -C 'type_e'")
        

        # exp_num = 111
        # os.system(f"sh train_ti_sdxl_with_regs.sh {exp} 0 0 {exp_num} 'ti_sdxl' -C 'type_e'")
        # exp_num = 112
        # os.system(f"sh train_ti_sdxl.sh {exp} {exp_num} 'ti_sdxl' -C 'type_e'")
        


        # all experiment under is for SD2
        stage_name ='CR'  

        # ti_emb_attn + db
        # check for 10 concepts
        exp_num = 0
        lambda_attn = 0.05
        lambda_emb = 1.5e-4
        # os.system(f"sh train_cr.sh {exp} 1 1 0 0 {exp_num} 'ti_emb_attn (orig CoRe)' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_emb_attn_DB (orig CoRe)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db
        # check for 10 concepts
        exp_num = 1
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db
        exp_num = 0
        lambda_attn = 0.05
        lambda_emb = 1.5e-4
        # os.system(f"sh train_cr.sh {exp} 1 1 0 0 {exp_num} 'ti_emb_attn (orig CoRe)' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_emb_attn_DB (orig CoRe)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # lambda_attn=10
        # lambda_gram=10
        # lambda_attn_noft=1e3
        # lambda_gram_noft=1e12

        # ti + db_attn
        exp_num = 11
        lambda_attn = 1e2
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 0 0 {exp_num} 'ti_DB_attn' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}") # also chp 500

        # ti + db_gram
        exp_num = 12
        lambda_gram = 10
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 0 0 {exp_num} 'ti_DB_gram' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_attn_noft
        exp_num = 13
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 1 0 {exp_num} 'ti_DB_attn_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}") # also chp 500

        # ti + db_gram_noft
        exp_num = 14
        lambda_gram_noft = 1e11
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 1 {exp_num} 'ti_DB_gram_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_gram_noft
        exp_num = 15
        lambda_gram_noft = 1e10
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 1 {exp_num} 'ti_DB_gram_noft_{lambda_gram_noft}' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_gram_noft
        exp_num = 16
        lambda_gram_noft = 1e12
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 1 {exp_num} 'ti_DB_gram_noft_{lambda_gram_noft}' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")


        ### Increasing the influence of the embedding reg
        # ti_emb + db
        exp_num = 17
        lambda_emb = 1
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 18
        lambda_emb = 10
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 19
        lambda_emb = 1e-1
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 20
        lambda_emb = 1e3
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 21
        lambda_emb = 1.5e-4
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 22
        lambda_emb = 1e8
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 27
        lambda_emb = 1e-3
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 28
        lambda_emb = 1e-2
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb + db
        exp_num = 29
        lambda_emb = 1e8
        # os.system(f"sh train_cr.sh {exp} 1 0 0 0 {exp_num} 'ti_emb_{lambda_emb}' {lambda_emb} -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")




        # ti + db_gram_noft
        # sqrt(loss_gram_noft), interpolate ca maps to max_seq
        exp_num = 23
        lambda_gram_noft = 1e5
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'ti_DB_gram_noft_{lambda_gram_noft}' {stage_name} {lambda_gram_noft} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_gram_noft
        # sqrt(loss_gram_noft), pairwise mse loss, not interpolate ca maps
        exp_num = 24
        lambda_gram_noft = 1e5
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'ti_DB_gram_noft_{lambda_gram_noft}' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_attn_noft
        # interpolate ca maps to max_seq
        exp_num = 25
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'ti_DB_attn_noft_{lambda_attn_noft}' {stage_name} {lambda_attn_noft} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_attn_noft
        # pairwise mse loss, not interpolate ca maps
        exp_num = 26
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'ti_DB_attn_noft_{lambda_attn_noft}' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # os.system("python run_vis_results.py")
        # os.system("python eval/eval/eval.py")


        # ti + db_attn
        exp_num = 2
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'ti_DB_attn' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_gram
        exp_num = 3
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'ti_DB_gram' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_attn_gram
        exp_num = 4
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 1 {exp_num} 'ti_DB_attn_gram' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db
        exp_num = 4
        # os.system(f"sh train_cr.sh {exp} 1 1 0 0 {exp_num} 'ti_emb_attn' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'ti_emb_attn_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db_attn
        exp_num = 5
        # os.system(f"sh train_cr.sh {exp} 1 1 0 0 {exp_num} 'ti_emb_attn' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'ti_emb_attn_DB_attn' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db_attn_noft
        exp_num = 6
        # os.system(f"sh train_cr.sh {exp} 1 1 0 0 {exp_num} 'ti_emb_attn' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'ti_emb_attn_DB_attn_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti_emb_attn + db_gram_noft
        exp_num = 7
        # os.system(f"sh train_cr.sh {exp} 1 1 0 0 {exp_num} 'ti_emb_attn' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'ti_emb_attn_DB_gram_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_attn_noft (pairwise loss)
        exp_num = 8
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'ti_DB_attn_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # ti + db_gram_noft (pairwise loss)
        exp_num = 9
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'ti_DB_gram_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # # ti + db_attn_noft_gram_noft
        exp_num = 10
        # os.system(f"sh train_cr.sh {exp} 0 0 0 0 {exp_num} 'ti' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 1 {exp_num} 'ti_DB_attn_noft_gram_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")





        stage_name ='CD'
        # cd + db
        exp_num = 1
        # os.system(f"sh train_cd.sh {exp} 0 0 0 {exp_num} 'cd' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_emb + db
        exp_num = 10
        lambda_emb = 1e3
        # os.system(f"sh train_cd.sh {exp} 1 0 0 {exp_num} 'cd_emb_{lambda_emb}' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_emb_{lambda_emb}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_attn_noft + db
        exp_num = 11
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cd.sh {exp} 0 1 0 {exp_num} 'cd_attn_noft_{lambda_attn_noft}' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_attn_noft_{lambda_attn_noft}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_gram_noft + db
        exp_num = 12
        lambda_gram_noft = 1e11
        # os.system(f"sh train_cd.sh {exp} 0 0 1 {exp_num} 'cd_gram_noft_{lambda_gram_noft}' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_gram_noft_{lambda_gram_noft}_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")



        # cd_gram_noft + db
        # sqrt(loss_gram_noft), interpolate ca maps to max_seq
        exp_num = 13
        lambda_gram_noft = 1e5
        # os.system(f"sh train_cd.sh {exp} 0 0 1 {exp_num} 'cd_gram_noft_{lambda_gram_noft}(sqrt(gram),interp_to_max)' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_gram_noft_{lambda_gram_noft}_DB (sqrt(gram), interp to max)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_gram_noft + db
        # sqrt(loss_gram_noft), pairwise mse loss, not interpolate ca maps
        exp_num = 14
        lambda_gram_noft = 1e5
        # os.system(f"sh train_cd.sh {exp} 0 0 1 {exp_num} 'cd_gram_noft_{lambda_gram_noft} (sqrt pairwise_loss)' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_gram_noft_{lambda_gram_noft}_DB (sqrt pairwise_loss)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_attn_noft + db
        # interpolate ca maps to max_seq
        exp_num = 15
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cd.sh {exp} 0 1 0 {exp_num} 'cd_attn_noft_{lambda_attn_noft} (interp to max)' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_attn_noft_{lambda_attn_noft}_DB (interp to max)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_attn_noft + db
        # pairwise mse loss, not interpolate ca maps
        exp_num = 16
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cd.sh {exp} 0 1 0 {exp_num} 'cd_attn_noft_{lambda_attn_noft} (pairwise_loss)' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_attn_noft_{lambda_attn_noft}_DB (pairwise_loss)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")



        # cd + db_attn_noft, pairwise loss
        exp_num = 8
        lambda_attn_noft = 1e3
        # os.system(f"sh train_cd.sh {exp} 0 0 0 {exp_num} 'cd' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'cd_DB_attn_noft_{lambda_attn_noft}' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd + db_gram_noft, sqrt pairwise loss
        exp_num = 9
        lambda_gram_noft = 1e5
        # os.system(f"sh train_cd.sh {exp} 0 0 0 {exp_num} 'cd' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'cd_DB_gram_noft_{lambda_gram_noft} (sqrt pairwise loss)' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_emb + db
        exp_num = 2
        lambda_emb = 1
        # os.system(f"sh train_cd.sh {exp} 1 0 0 {exp_num} 'cd_emb' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_emb_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_attn_noft + db
        exp_num = 3
        lambda_attn_noft = 10
        # os.system(f"sh train_cd.sh {exp} 0 1 0 {exp_num} 'cd_attn_noft' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_attn_noft_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_emb_attn_noft + db
        exp_num = 4
        # os.system(f"sh train_cd.sh {exp} 1 1 0 {exp_num} 'cd_emb_attn_noft' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_emb_attn_noft_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd_gram_noft + db
        exp_num = 5
        lambda_gram_noft = 1e8
        # os.system(f"sh train_cd.sh {exp} 0 0 1 {exp_num} 'cd_gram_noft' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 0 {exp_num} 'cd_gram_noft_DB' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd + db_attn_noft
        exp_num = 6
        # os.system(f"sh train_cd.sh {exp} 0 0 0 {exp_num} 'cd' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 1 0 {exp_num} 'cd_DB_attn_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        # cd + db_gram_noft
        exp_num = 7
        # os.system(f"sh train_cd.sh {exp} 0 0 0 {exp_num} 'cd' -C 'type_e'")
        # os.system(f"sh train_db.sh {exp} 0 1 {exp_num} 'cd_DB_gram_noft' {stage_name} -C 'type_e'")
        # os.system(f"sh inference_cd.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")




        # db (initilize to superclass)
        stage_name ='DB'
        exp_num = 1
        # os.system(f"sh train_db.sh cat cat cat cat live 0 0 {exp_num} 'DB_init_cls' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        exp_num = 2
        # os.system(f"sh train_db.sh clock clock clock clock object 0 0 {exp_num} 'DB_init_cls' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        exp_num = 3
        # os.system(f"sh train_db.sh berry_bowl bowl bowl bowl object 0 0 {exp_num} 'DB_init_cls' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")

        exp_num = 4
        # os.system(f"sh train_db.sh dog dog dog dog live 0 0 {exp_num} 'DB_init_cls' {stage_name} -C 'type_e'")
        # os.system(f"sh inference.sh 00{exp_num}-res-{concept}_{stage_name} {chp}")




    # dog6
    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 1 1 0 001 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 001 -C 'type_e'")

    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 1 0 0 002 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 002 -C 'type_e'")

    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 1 0 003 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog 006 -C 'type_e'")

    # os.system(f"sh train_cr.sh dog6 '<dog6>' dog dog live 0 0 0 1 0011 -C 'type_e'")
    # os.system(f"sh train_db.sh dog6 '<dog6>' dog dog live 1 1 011 -C 'type_e'")


if __name__ == '__main__':
    main()
