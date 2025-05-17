import os

def main():
    chp = 300
    num_of_res_pics = 10
    prompt_subset_length = 9
    
    exps = {
        "cat": "cat '<cat>' cat",
        # "dog6": "dog6 '<dog6>' dog",
        # "grey_sloth_plushie": "grey_sloth_plushie '<sloth-plushie>' toy",
        # "teapot": "teapot '<teapot>' teapot",
        # "shiny_sneaker": "shiny_sneaker '<sneaker>' sneaker",
        # "bear_plushie": "bear_plushie '<bear-plushie>' toy",
        # "colorful_sneaker": "colorful_sneaker '<sneaker>' sneaker",

        # "cat2": "cat2 '<cat2>' cat",
        # "duck_toy": "duck_toy '<duck_toy>' toy",
        # "clock": "clock '<clock>' clock",
        # "backpack": "backpack '<backpack>' backpack",
        # "berry_bowl": "berry_bowl '<bowl>' bowl",
        # "red_cartoon": "red_cartoon '<red_cartoon>' cartoon", 
        # "poop_emoji": "poop_emoji '<poop_emoji>' toy",

        # "dog": "dog '<dog>' dog",
        # "monster_toy": "monster_toy '<monster_toy>' toy",
        # "grey_sloth_plushie": "grey_sloth_plushie '<grey_sloth_plushie>' toy",
        # "vase": "vase '<vase>' vase",
        # "dog2": "dog2 '<dog2>' dog",
        # "dog5": "dog5 '<dog5>' dog",
    }

    for exp in exps.values():
        os.system(f"sh vis_results.sh {exp} {chp} {num_of_res_pics} {prompt_subset_length}")
            

if __name__ == '__main__':
    main()
