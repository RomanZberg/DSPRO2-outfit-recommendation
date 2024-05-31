from enum import IntEnum

accessories_en = [
    "ascot tie", "bandanna", "belt", "beret", "bib", "boa", "bonnet", "bow", "bow tie",
    "bracelet", "buckle", "button", "cap", "cuff", "cuff links", "earrings", "elastic",
    "fez", "glasses", "handbag", "handkerchief", "hat", "headscarf", "helmet", "hook and eye",
    "jewelry", "kerchief", "lapel", "lingerie", "loincloth", "muffler", "neckerchief",
    "necklace", "pocket", "pocketbook", "purse", "ring", "scarf", "shoulder pads",
    "snaps", "stockings", "sunglasses", "suspenders", "tie", "tie clip",
    "veil", "velcro", "visor", "wig", "zipper", "bag"
]

outerwear_en = [
    "anorak", "balaclava", "battledress", "blazer", "boa", "cape", "cardigan",
    "cloak", "coat", "coveralls", "cowboy hat", "cummerbund", "dashiki", "dinner jacket",
    "flak jacket", "fur coat", "gabardine", "gaiters", "gloves", "gown", "hazmat suit",
    "hoodie", "hospital gown", "housecoat", "jacket", "lab coat", "leather jacket",
    "life jacket", "overcoat", "overshirt", "parka", "pea coat", "raincoat", "robe",
    "shawl", "smock", "spacesuit", "suit", "sweater", "sweatshirt", "sweatsuit",
    "top coat", "trench coat", "windbreaker", "wrap"
]

bottomwear_en = [
    "bell-bottoms", "bermuda shorts", "bloomers", "breeches", "briefs", "capris", "chinos",
    "culottes", "drawers", "dungarees", "fatigues", "harem pants", "hose", "jodhpurs", "jumpsuit",
    "khakis", "kilt", "leggings", "pantyhose", "shorts", "skirt", "slacks", "sweatpants",
    "swim trunks", "trousers", "underwear", "tights"
]

shoes_en = [
    "boots", "clogs", "cowboy boots", "flip-flops", "galoshes", "loafers", "moccasins",
    "pumps", "sandals", "shoes", "slippers", "sneakers", "stilettos", "waders", "Wellingtons",
    "zoris"
]

accessories_de = [
    "fliege", "gürtel", "haube", "hut", "krawatte", "mütze", "quastenstola", "schal",
    "socken", "strümpfe", "winterschal", "xylographen-schutzbrille", "umhängetasche"
]

accessories_de.extend([
    "ascot-krawatte", "bandana", "gürtel", "baskenmütze", "lätzchen", "boa", "käppchen", "schleife", "fliege",
    "armband", "schnalle", "knopf", "kappe", "manschette", "manschettenknöpfe", "ohrringe", "gummi",
    "tarbusch", "brille", "handtasche", "taschentuch", "hut", "kopftuch", "helm", "haken und auge",
    "schmuck", "kopftuch", "revers", "dessous", "lendenschurz", "schal", "halstuch",
    "halskette", "tasche", "portemonnaie", "handtasche", "ring", "schal", "schulterpolster",
    "druckknöpfe", "socken", "strümpfe", "sonnenbrille", "träger", "krawatte", "krawattennadel",
    "schleier", "klettverschluss", "visier", "perücke", "reißverschluss", "tagesrucksack", "kopfhörer"
])

outerwear_de = [
    "anorak", "daunenjacke", "dufflecoat", "jacke", "jeansjacke",
    "mantel", "morgenmantel", "regenmantel", "regenponcho", "sakko",
    "schijacke", "trenchcoat", "wintermantel", "strickpullover", "kapuzenpullover", "bomberjacke", "weste",
    "winterjacke", "strickjacke", "kunstlederjacke", "fleecejacke", "softshelljacke", "kurzmantel"
]

innerwear_de = [
    "t-shirt", "longarm-shirt", "pullunder", "pullover", "sweater", "hoodie", "top", "langarmshirt",
]

bottomwear_de = [
    "cargohose", "chinohose", "hose", "jeggings", "jeans", "latzhose", "leggings",
    "shorts", "sportleggings", "treggings", "stoffhose", "jogginghose", "sporthose", "sweatjacke", "minirock",
    "lederhose", "schihose"
]

shoes_de = [
    "ballerinas", "chucks", "espadrilles", "flip-flops", "gummistiefel", "mokassins",
    "overknees", "sneaker", "stiefel", "stiefeletten", "velourslederboots", "winterschuhe", "laufschuh",
    "stiefelette", "plateaustiefelette", "plateaustiefel", "schnürstiefelette", "pantolette", "hausschuh",
    "hikingschuh",
    "sandalette", "cowboy-/bikerboot", "snowboot/winterstiefel", "skateschuh", "badesandale"
]

accessories = []
innerwear = []
outerwear = []
bottomwear = []
shoes = []

accessories.extend(accessories_de)
accessories.extend(accessories_en)

innerwear.extend(innerwear_de)

outerwear.extend(outerwear_de)
outerwear.extend(outerwear_en)

bottomwear.extend(bottomwear_de)
bottomwear.extend(bottomwear_en)

shoes.extend(shoes_de)
shoes.extend(shoes_en)


class WearType(IntEnum):
    accessoire = 1
    innerWear = 2
    outerWear = 3
    bottomWear = 4
    shoes = 5


def is_a_cloting_item_name_in_string(string, category_list):
    for clothing_item in category_list:
        if clothing_item in string.split(' '):
            return True


def get_clothing_item_type(string) -> WearType | None:
    if is_a_cloting_item_name_in_string(string, innerwear):
        return WearType.innerWear
    elif is_a_cloting_item_name_in_string(string, outerwear):
        return WearType.outerWear
    elif is_a_cloting_item_name_in_string(string, bottomwear):
        return WearType.bottomWear
    elif is_a_cloting_item_name_in_string(string, shoes):
        return WearType.shoes
    elif is_a_cloting_item_name_in_string(string, accessories):
        return WearType.accessoire
    else:
        return None
