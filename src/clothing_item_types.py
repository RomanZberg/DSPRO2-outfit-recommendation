from enum import IntEnum
from typing import List

accessories_en: List[str] = [
    "ascot tie", "bandanna", "belt", "beret", "bib", "boa", "bonnet", "bow", "bow tie",
    "bracelet", "button", "cap", "cuff", "cuff links", "earrings",
    "fez", "glasses", "handbag", "handkerchief", "hat", "headscarf", "helmet",
    "jewelry", "kerchief", "lapel", "loincloth", "muffler", "neckerchief",
    "necklace", "purse", "ring", "scarf", "shoulder pads",
    "snaps", "stockings", "sunglasses", "suspenders", "tie",
    "veil", "wig", "bag", "balaclava", "boa", "headband", "glove", "gloves", "clutch"
]

outerwear_en: List[str] = [
    "anorak", "battledress", "blazer", "cape", "cardigan",
    "cloak", "coat", "coveralls", "cowboy hat", "cummerbund", "dashiki", "dinner jacket",
    "flak jacket", "fur coat",  "gaiters", "gloves", "gown", "hazmat suit",
    "hoodie", "hospital gown", "housecoat", "jacket", "lab coat", "leather jacket",
    "life jacket", "overcoat", "overshirt", "parka", "pea coat", "raincoat", "robe",
    "shawl", "smock", "spacesuit", "suit", "sweater", "sweatshirt", "sweatsuit",
    "top coat", "trench coat", "windbreaker", "wrap", "pullover", "sweater", "hoodie"
]

innerwear_en: List[str] = [
    "t-shirt", "longsleeve", "top", "shirt",
    "undershirt", "tank top", "camisole", "thermal shirt",
    "sleeveless shirt", "bodysuit", "base layer", "sports bra",
    "crop top", "tube top", "singlet", "vest", "henley",
    "polo shirt", "muscle shirt", "jersey", "shirt", "bralette", "blouse", "shirt",
    "sleeve"
]

bottomwear_en: List[str] = [
    "bell-bottoms", "bermuda shorts", "bloomers", "breeches", "briefs", "capris", "chinos",
    "culottes", "drawers", "dungarees", "fatigues", "harem pants", "hose", "jodhpurs", "jumpsuit",
    "khakis", "kilt", "leggings", "pantyhose", "shorts", "skirt", "slacks", "sweatpants",
    "swim trunks", "trousers", "tights"
]

shoes_en: List[str] = [
    "boots", "clogs", "cowboy boots", "flip-flops", "galoshes", "loafers", "moccasins",
    "pumps", "sandals", "shoes", "slippers", "sneakers", "stilettos", "waders", "wellingtons",
    "zoris", "air force", "high heels", "boot", "loafer", "sneaker"
]

accessories_de: List[str] = [
    'helm', 'fes', 'schleier', 'visier', 'käppchen', 'sturmhaube', 'portemonnaie', 'tasche',
    'kappe', 'haube', 'armband', 'brille', 'ohrringe', 'mütze', 'boa', 'schmuck', 'kopftuch',
    'reissverschluss', 'kopfhörer', 'halstuch', 'schulterpolster',
    'krawattennadel', 'tagesrucksack', 'ring', 'knopf', 'halskette', 'fliege', 'druckknöpfe',
    'haken und auge', 'lätzchen', 'bandanna', 'gürtel', 'ascot tie', 'manschettenknöpfe',
    'lendenschurz', 'strümpfe', 'hosenträger', 'träger', 'handtasche', 'ascot-krawatte', 'schal',
    'perücke', 'geldbörse', 'schleife', 'revers', 'baskenmütze', 'bandanna', 'sonnenbrille', 'schnalle',
    'klettverschluss', 'krawatte', 'manschette', 'hut', 'taschentuch', 'dessous', "handtasche", "sonnenbrille",
    "umhängetasche"
]

outerwear_de: List[str] = [
    "anorak", "daunenjacke", "dufflecoat", "jacke", "jeansjacke", "lederjacke",
    "mantel", "morgenmantel", "regenmantel", "regenponcho", "sakko",
    "skijacke", "trenchcoat", "wintermantel", "strickpullover", "kapuzenpullover", "bomberjacke", "weste",
    "winterjacke", "strickjacke", "kunstlederjacke", "fleecejacke", "softshell jacke", "kurzmantel",
    "strickpullover"
]

innerwear_de: List[str] = [
    "t-shirt", "langarmshirt", "top", "hemd",
    "unterhemd", "tanktop", "trägertop", "thermohemd",
    "ärmelloses hemd", "body", "basislage", "sport-bh",
    "bauchfreies top", "bandeau", "singlet", "weste", "henley",
    "polohemd", "muskelshirt", "trikot", "bluse"
]

bottomwear_de: List[str] = [
    "cargohose", "chino hose", "hose", "jeggings", "jeans", "latzhose", "leggings",
    "shorts", "sportleggings", "treggings", "stoffhose", "jogginghose", "sporthose", "sweatjacke", "minirock",
    "lederhose", "skihose", "chinos", "stoffhose"
]

shoes_de: List[str] = [
    "ballerinas", "chucks", "espadrilles", "flip-flops", "gummistiefel", "mokassins",
    "overknees", "sneaker", "stiefel", "stiefeletten", "veloursleder boots", "winterschuhe", "laufschuh",
    "stiefelette", "plateaus tiefelette", "plateau stiefel", "schnürstiefelette", "pantolette", "hausschuh",
    "wanderschuhe",
    "sandalette", "cowboy-/bikerboot", "snowboot/winterstiefel", "skateschuh", "badesandale"
]

accessories: List[str] = []
innerwear: List[str] = []
outerwear: List[str] = []
bottomwear: List[str] = []
shoes: List[str] = []

accessories.extend(accessories_de)
accessories.extend(accessories_en)

innerwear.extend(innerwear_de)
innerwear.extend(innerwear_en)

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


def is_a_clothing_item_name_in_string(string: str, category_list: List[str]) -> bool:
    for clothing_item in category_list:
        if clothing_item in string.split(' '):
            return True


def get_clothing_item_type(string) -> WearType | None:
    if is_a_clothing_item_name_in_string(string, innerwear):
        return WearType.innerWear
    elif is_a_clothing_item_name_in_string(string, outerwear):
        return WearType.outerWear
    elif is_a_clothing_item_name_in_string(string, bottomwear):
        return WearType.bottomWear
    elif is_a_clothing_item_name_in_string(string, shoes):
        return WearType.shoes
    elif is_a_clothing_item_name_in_string(string, accessories):
        return WearType.accessoire
    else:
        return None
