from ram.models import ram_plus, ram

REGISTERED_RAM = {
    'ram_plus': ram_plus,
    'ram': ram
}

llm_tag_des_url = "datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json"

REGISTERED_RAM_MODEL: dict[str, str] = {
    'ram_plus': 'weights/ram/ram_plus_swin_large_14m.pth',
    'ram': 'weights/ram/ram_swin_large_14m.pth',
}
REGISTERED_VIT_MODEL = ['swin_base', 'swin_l']
