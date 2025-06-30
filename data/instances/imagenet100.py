import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.datasets as datasets

# name_dict = {'n01558993': 'robin', 'n01692333': 'Gila_monster', 'n01729322': 'hognose_snake', 'n01735189': 'garter_snake', 'n01749939': 'green_mamba', 'n01773797': 'garden_spider', 'n01820546': 'lorikeet', 'n01855672': 'goose', 'n01978455': 'rock_crab', 'n01980166': 'fiddler_crab', 'n01983481': 'American_lobster', 'n02009229': 'little_blue_heron', 'n02018207': 'American_coot', 'n02085620': 'Chihuahua', 'n02086240': 'Shih-Tzu', 'n02086910': 'papillon', 'n02087046': 'toy_terrier', 'n02089867': 'Walker_hound', 'n02089973': 'English_foxhound', 'n02090622': 'borzoi', 'n02091831': 'Saluki', 'n02093428': 'American_Staffordshire_terrier', 'n02099849': 'Chesapeake_Bay_retriever', 'n02100583': 'vizsla', 'n02104029': 'kuvasz', 'n02105505': 'komondor', 'n02106550': 'Rottweiler', 'n02107142': 'Doberman', 'n02108089': 'boxer', 'n02109047': 'Great_Dane', 'n02113799': 'standard_poodle', 'n02113978': 'Mexican_hairless', 'n02114855': 'coyote', 'n02116738': 'African_hunting_dog', 'n02119022': 'red_fox', 'n02123045': 'tabby', 'n02138441': 'meerkat', 'n02172182': 'dung_beetle', 'n02231487': 'walking_stick', 'n02259212': 'leafhopper', 'n02326432': 'hare', 'n02396427': 'wild_boar', 'n02483362': 'gibbon', 'n02488291': 'langur', 'n02701002': 'ambulance', 'n02788148': 'bannister', 'n02804414': 'bassinet', 'n02859443': 'boathouse', 'n02869837': 'bonnet', 'n02877765': 'bottlecap', 'n02974003': 'car_wheel', 'n03017168': 'chime', 'n03032252': 'cinema', 'n03062245': 'cocktail_shaker', 'n03085013': 'computer_keyboard', 'n03259280': 'Dutch_oven', 'n03379051': 'football_helmet', 'n03424325': 'gasmask', 'n03492542': 'hard_disc', 'n03494278': 'harmonica', 'n03530642': 'honeycomb', 'n03584829': 'iron', 'n03594734': 'jean', 'n03637318': 'lampshade', 'n03642806': 'laptop', 'n03764736': 'milk_can', 'n03775546': 'mixing_bowl', 'n03777754': 'modem', 'n03785016': 'moped', 'n03787032': 'mortarboard', 'n03794056': 'mousetrap', 'n03837869': 'obelisk', 'n03891251': 'park_bench', 'n03903868': 'pedestal', 'n03930630': 'pickup', 'n03947888': 'pirate', 'n04026417': 'purse', 'n04067472': 'reel', 'n04099969': 'rocking_chair', 'n04111531': 'rotisserie', 'n04127249': 'safety_pin', 'n04136333': 'sarong', 'n04229816': 'ski_mask', 'n04238763': 'slide_rule', 'n04336792': 'stretcher', 'n04418357': 'theater_curtain', 'n04429376': 'throne', 'n04435653': 'tile_roof', 'n04485082': 'tripod', 'n04493381': 'tub', 'n04517823': 'vacuum', 'n04589890': 'window_screen', 'n04592741': 'wing', 'n07714571': 'head_cabbage', 'n07715103': 'cauliflower', 'n07753275': 'pineapple', 'n07831146': 'carbonara', 'n07836838': 'chocolate_sauce', 'n13037406': 'gyromitra', 'n13040303': 'stinkhorn'}

class ImageNet100Dataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        self.dset = datasets.ImageFolder(os.path.join(root, f"ImageNet100/{split}"), transform=transform)
        self.targets = [item[1] for item in self.dset.imgs]
        self.class_to_idx = self.dset.class_to_idx
        self.class_names = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def get_num_classes(self):
        return len(self.class_names)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        file_name = Path(self.dset.imgs[index][0]).stem
        img, target = self.dset[index]
        return {"x": img, "y": target, "class_name": self.class_names[target], "file_name": file_name}