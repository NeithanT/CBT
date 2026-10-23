"""
Joint Balanced Dataset & Batch Encoder for ScaledGPT (Dual T4 Optimization).
Supports:
1. Multi-digit column scratchpad arithmetic (1 to 50+ digits curriculum).
2. Expanded tool repertoire:
   - Single-tool calling (calculator, weather, search, time, unit_convert)
   - Multi-tool sequential chaining
   - Nested JSON parameter schemas
   - Error handling and recovery traces
   - Multi-turn conversational dialogues
   - Direct conversational routing (no-tool)
3. Strict prompt and environment output loss masking.
"""

import os
import sys
import json
import random
from typing import List, Dict, Tuple, Any, Optional
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from champion_model.scaled_t4.tokenizer import ScaledTokenizer
except ImportError:
    from tokenizer import ScaledTokenizer


def format_addition_char_scratchpad(a: int, b: int) -> str:
    """Formats addition into column-by-column carry scratchpad for character tokenizer."""
    s_a, s_b = str(a), str(b)
    len_a, len_b = len(s_a), len(s_b)
    max_len = max(len_a, len_b)

    carry = 0
    steps, digits = [], []

    for col in range(max_len):
        da = int(s_a[len_a - 1 - col]) if col < len_a else 0
        db = int(s_b[len_b - 1 - col]) if col < len_b else 0
        col_sum = carry + da + db
        next_carry = col_sum // 10
        res_digit = col_sum % 10
        steps.append(f"({carry}+{da}+{db})={next_carry}{res_digit}")
        digits.append(str(res_digit))
        carry = next_carry

    if carry > 0:
        steps.append(f"({carry}+0+0)=0{carry}")
        digits.append(str(carry))

    ans = "".join(reversed(digits))
    return f"{a}+{b}=<THINK>" + "".join(steps) + f"</THINK>={ans}"


def extract_scratchpad_answer(text: str) -> Tuple[str, str]:
    """Extracts reasoning scratchpad and predicted answer from generated output."""
    reasoning = ""
    ans = ""
    if "<THINK>" in text and "</THINK>" in text:
        parts = text.split("</THINK>")
        reasoning = parts[0].replace("<THINK>", "")
        after = parts[1]
        if "=" in after:
            ans = after.split("=")[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
        else:
            ans = after.replace("<EOS>", "").replace("<PAD>", "").strip()
    elif "=" in text:
        ans = text.split("=")[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
    return reasoning, ans

# Factual databases for synthetic generation
CITIES = [
    "New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Madrid", 
    "Rome", "Seoul", "Singapore", "Zurich", "Honolulu", "Reykjavik", "Ouagadougou",
    "Vancouver", "Santiago", "Auckland", "Casablanca", "Vienna", "Stockholm", "Kyoto",
    "Dublin", "Oslo", "Helsinki", "Bangkok", "Cairo", "Lisbon", "Amsterdam", "Athens",
    "Buenos Aires", "Kuala Lumpur", "Rio de Janeiro", "San Francisco", "Hong Kong",
    "Los Angeles", "Mexico City", "Cape Town", "Sao Paulo",
]
CONDITIONS = ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy", "Foggy", "Clear"]
TIMEZONES = ["UTC", "EST", "PST", "GMT", "JST", "CET", "AEST", "CST", "IST", "HST"]

# Nonce word phonetic generators to force pure Attention Induction Copying
NONCE_ONSETS = ["bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "qu", "v", "z", "k", "j"]
NONCE_NUCLEI = ["a", "e", "i", "o", "u", "y", "ai", "ea", "ee", "oa", "oo", "ou"]
NONCE_CODAS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "z", "ck", "nk", "mp", "sh", "th", "nt", "rt"]


def generate_nonce_entity(rng: random.Random, allow_multi_word: bool = True) -> Tuple[str, str, str]:
    """Generates an arbitrary pseudo-word entity (singular, plural, type), supporting single and multi-word spans."""
    ent_type = rng.choice(["concept", "plant", "creature", "artifact", "substance", "person", "theory"])

    # 30% chance of multi-word compound nonce (2-3 words) to train multi-word pointer induction
    if allow_multi_word and rng.random() < 0.30:
        n_words = rng.choice([2, 2, 3])
        words = []
        for _ in range(n_words):
            n_syl = rng.choice([1, 2, 2, 3])
            w = "".join(rng.choice(NONCE_ONSETS) + rng.choice(NONCE_NUCLEI) + rng.choice(NONCE_CODAS) for _ in range(n_syl))
            words.append(w.capitalize())
        singular = " ".join(words)
        last_w = words[-1]
        plural_last = last_w + ("es" if last_w[-1] in "sxz" else "s")
        plural = " ".join(words[:-1] + [plural_last])
        return singular, plural, ent_type

    n_syl = rng.choice([1, 2, 2, 3, 4])
    parts = []
    for _ in range(n_syl):
        parts.append(rng.choice(NONCE_ONSETS) + rng.choice(NONCE_NUCLEI) + rng.choice(NONCE_CODAS))
    singular = "".join(parts)

    # 20% chance of complex morphological suffix to train long-token copy fidelity
    if rng.random() < 0.20:
        suffix = rng.choice(["ism", "ology", "ification", "ation", "ological", "ity"])
        singular = singular + suffix

    plural = singular + ("es" if singular[-1] in "sxz" else "s")
    return singular, plural, ent_type
SEARCH_FACTS = [
    ("Python creator", "Guido van Rossum", "creator"),
    ("capital of Australia", "Canberra", "capital"),
    ("speed of light", "299792 km/s", "constant"),
    ("tallest mountain on Earth", "Mount Everest", "geography"),
    ("deepest ocean trench", "Mariana Trench", "geography"),
    ("inventor of the telephone", "Alexander Graham Bell", "inventor"),
    ("currency of Japan", "Yen", "currency"),
    ("largest planet in solar system", "Jupiter", "astronomy"),
    ("author of Hamlet", "William Shakespeare", "literature"),
    ("chemical symbol for Gold", "Au", "chemistry"),
    ("Tokyo population", "14000000", "demographics"),
    ("Paris land area", "105 sq km", "geography"),
    ("distance from Earth to Moon", "384400 km", "astronomy"),
    ("human body temperature", "37 Celsius", "biology"),
]

# Open-domain entity database for Universal Search Delegation
SEARCH_ENTITIES = [
    # Foods & Plants
    {"entity": "tomato", "plural": "tomatoes", "type": "plant", "snippet": "the edible berry of the plant Solanum lycopersicum, commonly prepared as a culinary vegetable."},
    {"entity": "apple", "plural": "apples", "type": "fruit", "snippet": "a round, edible fruit produced by an apple tree, cultivated worldwide."},
    {"entity": "potato", "plural": "potatoes", "type": "vegetable", "snippet": "a starchy, tuberous root crop of the nightshade family Solanaceae."},
    {"entity": "coffee", "plural": "coffee beans", "type": "plant", "snippet": "a brewed beverage prepared from roasted coffee beans of flowering Coffea species."},
    {"entity": "banana", "plural": "bananas", "type": "fruit", "snippet": "an elongated, edible fruit produced by large herbaceous plants in the genus Musa."},
    {"entity": "onion", "plural": "onions", "type": "vegetable", "snippet": "a pungent vegetable that is the most widely cultivated species of the genus Allium."},
    {"entity": "wheat", "plural": "wheat grains", "type": "crop", "snippet": "a cereal grain cultivated worldwide as a major dietary staple."},
    {"entity": "rice", "plural": "rice grains", "type": "crop", "snippet": "the seed of the grass species Oryza sativa, a primary staple food for billions of people."},
    {"entity": "carrot", "plural": "carrots", "type": "vegetable", "snippet": "a root vegetable, typically orange, rich in beta-carotene and dietary fiber."},
    {"entity": "garlic", "plural": "garlic cloves", "type": "plant", "snippet": "a species of bulbous flowering plant in the genus Allium, used for culinary seasoning."},
    # Animals
    {"entity": "dog", "plural": "dogs", "type": "animal", "snippet": "a domesticated descendant of the wolf, widely kept as a companion and working animal."},
    {"entity": "cat", "plural": "cats", "type": "animal", "snippet": "a small, carnivorous mammal known for its agility, flexible body, and companionship with humans."},
    {"entity": "dolphin", "plural": "dolphins", "type": "animal", "snippet": "a highly intelligent marine mammal belonging to the cetacean order, known for echolocation."},
    {"entity": "elephant", "plural": "elephants", "type": "animal", "snippet": "the largest existing land animal, characterized by a long trunk, tusks, and large ear flaps."},
    {"entity": "penguin", "plural": "penguins", "type": "animal", "snippet": "an aquatic, flightless bird living almost exclusively in the Southern Hemisphere."},
    {"entity": "lion", "plural": "lions", "type": "animal", "snippet": "a large cat of the genus Panthera, native to Africa and India, known for social prides."},
    {"entity": "tiger", "plural": "tigers", "type": "animal", "snippet": "the largest living cat species, identifiable by its reddish-orange coat with dark vertical stripes."},
    {"entity": "whale", "plural": "whales", "type": "animal", "snippet": "a widely distributed and diverse group of fully aquatic placental marine mammals."},
    # Science & Nature
    {"entity": "gravity", "plural": "gravitational forces", "type": "physics", "snippet": "a fundamental interaction which causes mutual attraction between all things with mass or energy."},
    {"entity": "photosynthesis", "plural": "photosynthetic processes", "type": "biology", "snippet": "a biological process used by plants to convert light energy into chemical energy stored in glucose."},
    {"entity": "atom", "plural": "atoms", "type": "physics", "snippet": "the basic unit of a chemical element, consisting of a dense central nucleus surrounded by electrons."},
    {"entity": "DNA", "plural": "DNA molecules", "type": "genetics", "snippet": "deoxyribonucleic acid, a polymer composed of polynucleotide chains carrying genetic instructions."},
    {"entity": "black hole", "plural": "black holes", "type": "astronomy", "snippet": "a region of spacetime where gravity is so strong that nothing, including light, can escape from it."},
    {"entity": "water", "plural": "water molecules", "type": "chemistry", "snippet": "a transparent, odorless chemical substance with chemical formula H2O, essential for life."},
    {"entity": "oxygen", "plural": "oxygen molecules", "type": "chemistry", "snippet": "a chemical element with symbol O and atomic number 8, essential for aerobic respiration."},
    {"entity": "electricity", "plural": "electric currents", "type": "physics", "snippet": "physical phenomena associated with the presence and motion of matter possessing electric charge."},
    # Geography & Landmarks
    {"entity": "Eiffel Tower", "plural": "monuments", "type": "landmark", "snippet": "a wrought-iron lattice tower on the Champ de Mars in Paris, France, named after Gustave Eiffel."},
    {"entity": "Great Wall of China", "plural": "ancient walls", "type": "history", "snippet": "a series of fortifications built across the historical northern borders of ancient Chinese states."},
    {"entity": "Mount Everest", "plural": "peaks", "type": "geography", "snippet": "Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas."},
    {"entity": "Amazon River", "plural": "rivers", "type": "geography", "snippet": "the largest river by discharge volume of water in the world, located in South America."},
    {"entity": "Sahara Desert", "plural": "deserts", "type": "geography", "snippet": "the largest hot desert in the world and the third largest desert overall, covering North Africa."},
    # Technology & Society
    {"entity": "computer", "plural": "computers", "type": "technology", "snippet": "an electronic device for storing and processing data, typically according to binary instructions."},
    {"entity": "internet", "plural": "computer networks", "type": "technology", "snippet": "a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP)."},
    {"entity": "algorithm", "plural": "algorithms", "type": "computer science", "snippet": "a finite sequence of rigorous instructions used to solve problems or perform a computation."},
    {"entity": "telescope", "plural": "telescopes", "type": "instrument", "snippet": "an optical instrument that makes distant objects appear magnified using lenses or curved mirrors."},
    {"entity": "airplane", "plural": "airplanes", "type": "vehicle", "snippet": "a powered, fixed-wing aircraft that is propelled forward by thrust from an engine."},
    # Historical Figures & Creators
    {"entity": "Guido van Rossum", "plural": "programmers", "type": "person", "snippet": "the Dutch programmer best known as the creator of the Python programming language."},
    {"entity": "Albert Einstein", "plural": "physicists", "type": "person", "snippet": "a theoretical physicist who developed the theory of relativity, a pillar of modern physics."},
    {"entity": "William Shakespeare", "plural": "playwrights", "type": "person", "snippet": "an English playwright, poet, and actor widely regarded as the greatest writer in the English language."},
    {"entity": "Alexander Graham Bell", "plural": "inventors", "type": "person", "snippet": "an inventor and scientist credited with patenting the first practical telephone."},
    {"entity": "Isaac Newton", "plural": "scientists", "type": "person", "snippet": "an English mathematician and physicist who formulated the laws of motion and universal gravitation."},
    {"entity": "Nikola Tesla", "plural": "engineers", "type": "person", "snippet": "a Serbian-American engineer known for his contributions to alternating current electricity."},
    {"entity": "Marie Curie", "plural": "scientists", "type": "person", "snippet": "a Polish-French physicist who pioneered radioactivity research and won two Nobel Prizes."},
    {"entity": "Alan Turing", "plural": "mathematicians", "type": "person", "snippet": "an English mathematician widely considered the father of modern computer science and AI."},
    {"entity": "Ada Lovelace", "plural": "mathematicians", "type": "person", "snippet": "an English mathematician celebrated for writing the first computer algorithm."},
    {"entity": "Aristotle", "plural": "philosophers", "type": "person", "snippet": "an ancient Greek philosopher whose work shaped centuries of Western logic and metaphysics."},
    {"entity": "Leonardo da Vinci", "plural": "polymaths", "type": "person", "snippet": "an Italian polymath active as a painter, draughtsman, engineer, and scientist."},
    # Additional Produce & Animals
    {"entity": "lemon", "plural": "lemons", "type": "fruit", "snippet": "a species of small evergreen tree native to Asia, known for its yellow ellipsoidal citrus fruit."},
    {"entity": "strawberry", "plural": "strawberries", "type": "fruit", "snippet": "a widely grown hybrid species of the genus Fragaria, cultivated for its sweet red fruit."},
    {"entity": "grape", "plural": "grapes", "type": "fruit", "snippet": "a botanical berry of the deciduous woody vines of the flowering plant genus Vitis."},
    {"entity": "spinach", "plural": "spinach leaves", "type": "vegetable", "snippet": "an edible flowering plant in the family Amaranthaceae, rich in iron and vitamins."},
    {"entity": "broccoli", "plural": "broccoli florets", "type": "vegetable", "snippet": "an edible green plant in the cabbage family whose large flower head is eaten as a vegetable."},
    {"entity": "avocado", "plural": "avocados", "type": "fruit", "snippet": "a medium-sized, evergreen tree native to the Americas, harvested for its buttery fruit."},
    {"entity": "wolf", "plural": "wolves", "type": "animal", "snippet": "a large canine native to Eurasia and North America, known for pack hunting behavior."},
    {"entity": "eagle", "plural": "eagles", "type": "animal", "snippet": "a large, powerfully built bird of prey belonging to the family Accipitridae."},
    {"entity": "shark", "plural": "sharks", "type": "animal", "snippet": "a group of elasmobranch fish characterized by a cartilaginous skeleton and multiple gill slits."},
    {"entity": "octopus", "plural": "octopuses", "type": "animal", "snippet": "a soft-bodied, eight-limbed mollusc of the order Octopoda, famous for high intelligence."},
    # Science & Computing
    {"entity": "quantum entanglement", "plural": "entangled states", "type": "physics", "snippet": "a physical phenomenon occurring when pairs of particles interact such that quantum states cannot be described independently."},
    {"entity": "thermodynamics", "plural": "thermodynamic laws", "type": "physics", "snippet": "a branch of physics that deals with heat, work, temperature, and statistical mechanics."},
    {"entity": "compiler", "plural": "compilers", "type": "computer science", "snippet": "a computer program that translates computer code written in one language into another."},
    {"entity": "operating system", "plural": "operating systems", "type": "computer science", "snippet": "system software that manages computer hardware, software resources, and common services."},
    {"entity": "neural network", "plural": "neural networks", "type": "computer science", "snippet": "a computational model inspired by the biological neural networks of animal brains."},
    # Alphanumeric, Scientific & Tech Entities
    {"entity": "CRISPR Cas9", "plural": "CRISPR Cas9 systems", "type": "genetics", "snippet": "a gene editing technology using an RNA-guided endonuclease enzyme."},
    {"entity": "Boeing 747", "plural": "Boeing 747 aircraft", "type": "aerospace", "snippet": "a wide-body commercial airliner and cargo transport aircraft."},
    {"entity": "Apollo 11", "plural": "Apollo missions", "type": "spaceflight", "snippet": "the American spaceflight that first landed humans on the Moon in 1969."},
    {"entity": "Web3", "plural": "decentralized protocols", "type": "technology", "snippet": "an idea for an iteration of the World Wide Web based on blockchain technology."},
    {"entity": "Formula 1", "plural": "Formula 1 races", "type": "motorsport", "snippet": "the highest class of international racing for open-wheel single-seater formula cars."},
    {"entity": "IPv6", "plural": "IPv6 addresses", "type": "networking", "snippet": "the most recent version of the Internet Protocol providing a vast numerical address space."},
    # Classical, Philosophical & Cultural Terms
    {"entity": "Marcus Aurelius", "plural": "Roman emperors", "type": "person", "snippet": "a Roman emperor and Stoic philosopher who ruled from 161 to 180 AD."},
    {"entity": "brutalist architecture", "plural": "brutalist structures", "type": "architecture", "snippet": "an architectural style characterized by minimalist constructions showcasing bare building materials."},
    {"entity": "existentialism", "plural": "existentialist doctrines", "type": "philosophy", "snippet": "a philosophical inquiry exploring human existence, freedom, and meaning."},
    {"entity": "Ragnarok", "plural": "mythological battles", "type": "mythology", "snippet": "in Norse mythology, a series of catastrophic events leading to the renewal of the world."},
    {"entity": "theremin", "plural": "theremins", "type": "instrument", "snippet": "an electronic musical instrument controlled without physical contact by metal antennas."},
    {"entity": "hyperinflation", "plural": "hyperinflationary cycles", "type": "economics", "snippet": "rapid, excessive, and out-of-control general price increases in an economy."},
    {"entity": "quantum physics", "plural": "quantum systems", "type": "physics", "snippet": "a fundamental branch of physics that studies nature at the scale of atoms and subatomic particles."},
    {"entity": "interstellar", "plural": "interstellar regions", "type": "astronomy", "snippet": "the matter and radiation that exists in the space between the star systems in a galaxy."},
    {"entity": "tiramisu", "plural": "tiramisus", "type": "cuisine", "snippet": "a coffee-flavoured Italian dessert made of ladyfingers dipped in coffee and layered with mascarpone."},
    {"entity": "neutron stars", "plural": "neutron stars", "type": "astronomy", "snippet": "the collapsed core of a massive supergiant star composed almost entirely of neutrons."},
    {"entity": "transformers in deep learning", "plural": "transformer architectures", "type": "computer science", "snippet": "a deep learning architecture based on multi-head self-attention mechanisms."},
]
DEPARTMENTS = ["engineering", "marketing", "sales", "finance", "legal", "product"]


# -----------------------------------------------------------------------------
# 1. Search Delegation & Copy Induction Dataset Generator
# -----------------------------------------------------------------------------

def generate_search_delegation_dataset(
    n_samples: int = 15000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generates concentrated Search Delegation samples for Induction Head copy training.
    Curriculum:
    - 65% Nonce Words (1-3 syllables) forcing pure Attention Pointer Copying (memorization impossible).
    - 35% Real-World Entities (tomatoes, photosynthesis, etc.) establishing lexical alignment.
    - Symmetric punctuation curriculum (50% with '?', '.', 50% raw text without punctuation).
    - Diverse question framing: What are, What is a, Define, Tell me about, Search for, Who is, Explain.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, str]] = []

    for _ in range(n_samples):
        is_nonce = (rng.random() < 0.65)
        has_punct = (rng.random() < 0.50)

        if is_nonce:
            ent, plural, ent_type = generate_nonce_entity(rng)
            snippet = f"a hypothetical {ent_type} characterized in verified theoretical studies."
        else:
            ent_item = rng.choice(SEARCH_ENTITIES)
            ent = ent_item["entity"]
            plural = ent_item.get("plural", ent + "s")
            snippet = ent_item["snippet"]
            ent_type = ent_item.get("type", "concept")

        # Choose query framing
        frame = rng.choice([
            "what_are_plural",
            "what_is_singular",
            "what_is_concept",
            "define",
            "tell_me_about",
            "search_for",
            "who_is",
            "explain",
            "lookup",
        ])

        if frame == "what_are_plural":
            q_mark = "?" if has_punct else ""
            user_prompt = f"What are {plural}{q_mark}"
            search_q = plural
            thought = rng.choice([
                "The user is asking a factual question. I will query the search tool.",
                "The user requests factual information. Querying search.",
                "The user is seeking knowledge on this topic. I will search the web.",
            ])
            answer = f"{plural.capitalize()} are {snippet}"
        elif frame == "what_is_singular" and ent_type in ["plant", "fruit", "vegetable", "animal", "creature", "artifact", "instrument", "vehicle"]:
            article = "an" if ent[0].lower() in "aeiou" else "a"
            q_mark = "?" if has_punct else ""
            user_prompt = f"What is {article} {ent}{q_mark}"
            search_q = ent
            thought = rng.choice([
                "The user is asking a factual question. I will query the search tool.",
                "The user is inquiring about this entity. I will query the search tool.",
                "The user requests factual information. Querying search.",
            ])
            answer = f"{ent.capitalize()} is {snippet}"
        elif frame == "define":
            p_mark = "." if has_punct else ""
            user_prompt = f"Define {ent}{p_mark}"
            search_q = ent
            thought = rng.choice([
                "The user asks for a definition. Calling search tool.",
                "The user is asking a factual question. I will query the search tool.",
                "The user requests factual information. Querying search.",
            ])
            answer = f"{ent.capitalize()} is defined as {snippet}"
        elif frame == "tell_me_about":
            p_mark = "." if has_punct else ""
            user_prompt = f"Tell me about {ent}{p_mark}"
            search_q = ent
            thought = rng.choice([
                "The user requests factual background information. Querying search.",
                "The user is asking a factual question. I will query the search tool.",
                "The user is seeking knowledge on this topic. I will search the web.",
            ])
            answer = f"According to verified search results, {ent} is {snippet}"
        elif frame == "search_for":
            p_mark = "." if has_punct else ""
            user_prompt = f"Search for {ent}{p_mark}"
            search_q = ent
            thought = rng.choice([
                "The user explicitly requests a search. Calling search tool.",
                "The user requests factual information. Querying search.",
                "The user is asking a factual question. I will query the search tool.",
            ])
            answer = f"Search results indicate {ent} is {snippet}"
        elif frame == "who_is" and ent_type in ["person", "creature"]:
            q_mark = "?" if has_punct else ""
            user_prompt = f"Who was {ent}{q_mark}" if rng.random() < 0.5 else f"Who is {ent}{q_mark}"
            search_q = ent
            thought = rng.choice([
                "The user requests biographical information. Calling search.",
                "The user is asking a factual question. I will query the search tool.",
                "The user requests factual information. Querying search.",
            ])
            answer = f"{ent} was {snippet}" if "was" in user_prompt else f"{ent} is {snippet}"
        elif frame == "explain":
            p_mark = "." if has_punct else ""
            user_prompt = f"Explain {ent}{p_mark}"
            search_q = ent
            thought = rng.choice([
                "The user asks for an explanation of this subject. I will query the search tool.",
                "The user is asking a factual question. I will query the search tool.",
                "The user requests factual information. Querying search.",
            ])
            answer = f"{ent.capitalize()} is {snippet}"
        else:
            q_mark = "?" if has_punct else ""
            user_prompt = f"What is {ent}{q_mark}"
            search_q = ent
            thought = rng.choice([
                "The user is asking a factual question. I will query the search tool.",
                "The user requests factual information. Querying search.",
                "The user is seeking knowledge on this topic. I will search the web.",
            ])
            answer = f"{ent.capitalize()} is {snippet}"

        call = f"search(query=\"{search_q}\")"
        result_json = json.dumps({"snippet": snippet})
        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought}</THINK>"
            f"<CALL>{call}</CALL>"
            f"<RESULT>{result_json}</RESULT>"
            f"<ANSWER>{answer}</ANSWER>"
        )

        samples.append({
            "task_type": "tool",
            "category": "search_delegation",
            "prompt": f"<USER>{user_prompt}</USER>",
            "full_text": full_text,
        })

    return samples


# -----------------------------------------------------------------------------
# 1. High-Digit Arithmetic Dataset Generator
# -----------------------------------------------------------------------------

def generate_arithmetic_equations(
    n_samples: int = 4000,
    seed: int = 42,
    max_train_digits: int = 15,
) -> List[Dict[str, str]]:
    """
    Generates balanced multi-digit addition equations with column carry scratchpad.
    Curriculum: 1 to max_train_digits (default 15) digits, plus edge cases:
    - Zero padding
    - Cascade carries (e.g. 9999 + 1)
    - Equal length and asymmetric lengths
    """
    rng = random.Random(seed)
    samples: List[Dict[str, str]] = []

    for _ in range(n_samples):
        mode = rng.random()

        if mode < 0.20:
            # Cascade carry stress cases (999... + 1 or 999... + small)
            digits = rng.randint(2, max_train_digits)
            a = int("9" * digits)
            b = rng.randint(1, 9)
        elif mode < 0.35:
            # Round numbers / power of 10 zero tracking
            d1 = rng.randint(2, max_train_digits)
            d2 = rng.randint(2, max_train_digits)
            a = rng.randint(1, 9) * (10 ** (d1 - 1))
            b = rng.randint(1, 9) * (10 ** (d2 - 1))
        elif mode < 0.60:
            # Uniform length random digits (1 to max_train_digits)
            d = rng.randint(1, max_train_digits)
            min_v = 10 ** (d - 1) if d > 1 else 0
            max_v = (10 ** d) - 1
            a = rng.randint(min_v, max_v)
            b = rng.randint(min_v, max_v)
        else:
            # Asymmetric length random digits
            d1 = rng.randint(1, max_train_digits)
            d2 = rng.randint(1, max_train_digits)
            min_v1 = 10 ** (d1 - 1) if d1 > 1 else 0
            max_v1 = (10 ** d1) - 1
            min_v2 = 10 ** (d2 - 1) if d2 > 1 else 0
            max_v2 = (10 ** d2) - 1
            a = rng.randint(min_v1, max_v1)
            b = rng.randint(min_v2, max_v2)

        full_text = format_addition_char_scratchpad(a, b)
        prompt = full_text.split("<THINK>")[0]
        samples.append({
            "task_type": "math",
            "prompt": prompt,
            "full_text": full_text,
        })

    return samples


# -----------------------------------------------------------------------------
# 2. Expanded Tool Repertoire Generator
# -----------------------------------------------------------------------------

def generate_tool_example(rng: random.Random, category: Optional[str] = None) -> Dict[str, str]:
    """
    Generates a structured synthetic tool interaction from 6 expanded categories:
    1. Single-tool calling (calculator, weather, search, time, unit_convert)
    2. Multi-tool sequential chaining
    3. Nested JSON parameter schemas
    4. Error handling and recovery
    5. Multi-turn dialogues
    6. Direct conversational routing (no-tool)

    Enforces 1-Hop Induction Copying:
    - <THINK> emits a clean, entity-agnostic routing intent (no variable copying in <THINK>).
    - <CALL> copies arguments directly from <USER> via 1-hop attention pointers.
    - Symmetric punctuation curriculum (50% with '?'/'.', 50% raw conversational text).
    `category` forces a specific category instead of sampling one.
    """
    category = category or rng.choices(
        [
            "single_tool",
            "multi_tool_chain",
            "nested_json_schema",
            "error_recovery",
            "multi_turn",
            "no_tool",
        ],
        weights=[0.55, 0.12, 0.12, 0.08, 0.06, 0.07],
        k=1,
    )[0]
    has_punct = (rng.random() < 0.50)
    q_mark = "?" if has_punct else ""
    p_mark = "." if has_punct else ""

    if category == "single_tool":
        sub = rng.choices(
            ["weather", "calculator", "time", "unit_convert", "search"],
            weights=[0.28, 0.24, 0.20, 0.20, 0.08],
            k=1,
        )[0]
        if sub == "calculator":
            op = rng.choice(["+", "-", "*", "/"])
            if op == "+":
                a, b = rng.randint(10, 9999), rng.randint(10, 9999)
                res = a + b
            elif op == "-":
                a, b = rng.randint(100, 9999), rng.randint(10, 99)
                res = a - b
            elif op == "*":
                a, b = rng.randint(2, 250), rng.randint(2, 99)
                res = a * b
            else:
                b = rng.randint(2, 50)
                res = rng.randint(2, 200)
                a = res * b

            use_spaces = (rng.random() < 0.50)
            expr_prompt = f"{a} {op} {b}" if use_spaces else f"{a}{op}{b}"
            # Keep call expression identical to prompt expression or normalized so 1-hop pointer copies cleanly
            call_expr = expr_prompt if rng.random() < 0.65 else f"{a}{op}{b}"
            user_prompt = rng.choice([
                f"Calculate {expr_prompt}{p_mark}",
                f"What is {expr_prompt}{q_mark}",
                f"Compute {expr_prompt}{p_mark}",
                f"Evaluate {expr_prompt}{p_mark}",
                f"Evaluate the mathematical expression {expr_prompt}{p_mark}",
                f"Solve {expr_prompt}{p_mark}",
            ])
            thought = rng.choice([
                "The user requests an arithmetic calculation. I will invoke the calculator tool.",
                "The user is asking to evaluate a mathematical expression. Calling calculator.",
            ])
            call = f"calculator(expr=\"{call_expr}\")"
            result = json.dumps({"result": res})
            answer = f"The evaluated result of {call_expr} is {res}."

        elif sub == "weather":
            if rng.random() < 0.35:
                # Nonce city to force pure copy induction (single or multi-word)
                nonce_city, _, _ = generate_nonce_entity(rng, allow_multi_word=True)
                city = " ".join(w.capitalize() for w in nonce_city.split())
            else:
                city = rng.choice(CITIES)
            temp = rng.randint(-5, 35)
            cond = rng.choice(CONDITIONS)
            user_prompt = rng.choice([
                f"What is the weather in {city}{q_mark}",
                f"What is the weather on {city}{q_mark}",
                f"What is the weather for {city}{q_mark}",
                f"What is the current weather forecast for {city}{q_mark}",
                f"What is the weather like in {city}{q_mark}",
                f"How is the weather in {city}{q_mark}",
                f"How is the weather on {city}{q_mark}",
                f"How is the weather for {city}{q_mark}",
                f"Weather in {city}{p_mark}",
                f"Weather on {city}{p_mark}",
                f"Weather for {city}{p_mark}",
                f"Check the weather in {city}{p_mark}",
                f"Check the weather for {city}{p_mark}",
                f"Tell me the weather in {city}{p_mark}",
                f"Forecast for {city}{p_mark}",
                f"{city} weather{p_mark}",
            ])
            thought = rng.choice([
                "The user requests current weather information. I will query the get_weather tool.",
                "The user is asking for weather conditions. Querying get_weather.",
            ])
            call = f"get_weather(city=\"{city}\")"
            result = json.dumps({"temp": temp, "condition": cond, "units": "celsius"})
            answer = f"The weather in {city} is currently {cond} with a temperature of {temp} degrees Celsius."

        elif sub == "search":
            search_mode = rng.random()
            if search_mode < 0.50:
                ent, plural, ent_type = generate_nonce_entity(rng, allow_multi_word=True)
                snippet = f"a hypothetical {ent_type} characterized in general knowledge."
            elif search_mode < 0.85 and bool(SEARCH_ENTITIES):
                ent_item = rng.choice(SEARCH_ENTITIES)
                ent = ent_item["entity"]
                plural = ent_item.get("plural", ent + "s")
                snippet = ent_item["snippet"]
                ent_type = ent_item.get("type", "concept")
            else:
                ent = None

            if ent is not None:
                variant = rng.choice([
                    "what_are_plural",
                    "what_is_a",
                    "what_is",
                    "tell_me_about",
                    "explain",
                    "define",
                    "search_for",
                    "lookup",
                    "who_is",
                ])

                if variant == "what_are_plural":
                    user_prompt = f"What are {plural}{q_mark}"
                    search_q = plural
                    answer = f"{plural.capitalize()} are {snippet}"
                elif variant == "what_is_a" and ent_type in ["plant", "fruit", "vegetable", "animal", "creature", "artifact", "instrument", "vehicle"]:
                    article = "an" if ent[0].lower() in "aeiou" else "a"
                    user_prompt = f"What is {article} {ent}{q_mark}"
                    search_q = ent
                    answer = f"{ent.capitalize()} is {snippet}"
                elif variant == "tell_me_about":
                    user_prompt = f"Tell me about {ent}{p_mark}"
                    search_q = ent
                    answer = f"According to verified search results, {ent} is {snippet}"
                elif variant == "define":
                    user_prompt = f"Define {ent}{p_mark}"
                    search_q = ent
                    answer = f"{ent.capitalize()} is defined as {snippet}"
                elif variant == "explain":
                    user_prompt = f"Explain {ent}{p_mark}"
                    search_q = ent
                    answer = f"{ent.capitalize()} is {snippet}"
                elif variant == "search_for":
                    user_prompt = f"Search for {ent}{p_mark}"
                    search_q = ent
                    answer = f"Search results indicate {ent} is {snippet}"
                elif variant == "lookup":
                    user_prompt = f"Look up {ent}{p_mark}"
                    search_q = ent
                    answer = f"Information for {ent}: {snippet}"
                elif variant == "who_is" and ent_type in ["person", "creature"]:
                    prefix = "Who was" if rng.random() < 0.5 else "Who is"
                    user_prompt = f"{prefix} {ent}{q_mark}"
                    search_q = ent
                    answer = f"{ent} is {snippet}"
                else:
                    user_prompt = f"What is {ent}{q_mark}"
                    search_q = ent
                    answer = f"{ent.capitalize()} is {snippet}"

                thought = rng.choice([
                    "The user is asking a factual question. I will query the search tool.",
                    "The user requests factual information. Querying search.",
                    "The user is seeking knowledge on this topic. I will search the web.",
                ])
                call = f"search(query=\"{search_q}\")"
                result = json.dumps({"snippet": snippet})
            else:
                topic, fact, _ = rng.choice(SEARCH_FACTS)
                user_prompt = rng.choice([
                    f"Who or what is {topic}{q_mark}",
                    f"What is {topic}{q_mark}",
                    f"Tell me about {topic}{p_mark}",
                ])
                thought = "The user is asking a factual question. I will query the search tool."
                call = f"search(query=\"{topic}\")"
                result = json.dumps({"snippet": fact})
                answer = f"According to verified search results, {topic} is {fact}."

        elif sub == "time":
            tz = rng.choice(TIMEZONES)
            hr = rng.randint(0, 23)
            mn = rng.randint(0, 59)
            time_str = f"{hr:02d}:{mn:02d}"
            user_prompt = rng.choice([
                f"What is the current time in {tz}{q_mark}",
                f"What is the time in {tz}{q_mark}",
                f"What time is it in {tz}{q_mark}",
                f"Current time in {tz}{p_mark}",
                f"Time in {tz}{p_mark}",
                f"Tell me the time in {tz}{p_mark}",
                f"Check the time for {tz}{p_mark}",
            ])
            thought = rng.choice([
                "The user requests the current time for a timezone. I will query the get_time tool.",
                "The user is asking for the current time. Querying get_time.",
            ])
            call = f"get_time(timezone=\"{tz}\")"
            result = json.dumps({"time": time_str, "timezone": tz})
            answer = f"The current time in {tz} is {time_str}."

        else: # unit_convert
            val = rng.randint(5, 500)
            from_u, to_u, factor = rng.choice([
                ("km", "miles", 0.621371),
                ("miles", "km", 1.60934),
                ("kg", "lbs", 2.20462),
                ("lbs", "kg", 0.453592),
            ])
            converted = round(val * factor, 2)
            user_prompt = rng.choice([
                f"Convert {val} {from_u} to {to_u}{p_mark}",
                f"How many {to_u} is {val} {from_u}{q_mark}",
                f"How many {to_u} are in {val} {from_u}{q_mark}",
                f"What is {val} {from_u} in {to_u}{q_mark}",
                f"Change {val} {from_u} into {to_u}{p_mark}",
            ])
            thought = rng.choice([
                "The user requests a unit conversion. I will invoke the unit_convert tool.",
                "The user is asking to convert between measurement units. Calling unit_convert.",
            ])
            call = f"unit_convert(value={val}, from_unit=\"{from_u}\", to_unit=\"{to_u}\")"
            result = json.dumps({"value": converted, "unit": to_u})
            answer = f"{val} {from_u} is equal to {converted} {to_u}."

        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought}</THINK>"
            f"<CALL>{call}</CALL>"
            f"<RESULT>{result}</RESULT>"
            f"<ANSWER>{answer}</ANSWER>"
        )

    elif category == "multi_tool_chain":
        # Chained sequential reasoning: Step 1 -> Tool 1 -> Result 1 -> Step 2 -> Tool 2 -> Result 2 -> Final Answer
        chain_type = rng.choice(["fact_calc", "dual_weather", "tz_calc"])
        if chain_type == "fact_calc":
            pop = 14000000
            area = 2194
            density = round(pop / area, 2)
            user_prompt = f"What is the population density of Tokyo based on its population and area of 2194 sq km{q_mark}"
            thought1 = "First, I need to search for Tokyo's population."
            call1 = "search(query=\"Tokyo population\")"
            result1 = json.dumps({"population": pop})
            thought2 = "Now I will calculate the population density using the calculator tool."
            call2 = f"calculator(expr=\"{pop} / {area}\")"
            result2 = json.dumps({"result": density})
            answer = f"Tokyo's population is {pop:,}. Dividing by 2,194 sq km gives a density of approximately {density} people per sq km."

        elif chain_type == "dual_weather":
            c1, c2 = rng.sample(CITIES, 2)
            t1, t2 = rng.randint(0, 30), rng.randint(0, 30)
            diff = abs(t1 - t2)
            user_prompt = rng.choice([
                f"Compare the temperature between {c1} and {c2}{p_mark}",
                f"Compare the weather in {c1} and {c2}{p_mark}",
                f"What is the temperature difference between {c1} and {c2}{q_mark}",
            ])
            thought1 = "I will first query the weather for the first city."
            call1 = f"get_weather(city=\"{c1}\")"
            result1 = json.dumps({"temp": t1, "condition": "Cloudy"})
            thought2 = "Next, I will query the weather for the second city."
            call2 = f"get_weather(city=\"{c2}\")"
            result2 = json.dumps({"temp": t2, "condition": "Sunny"})
            answer = f"{c1} is {t1}C and {c2} is {t2}C, with a temperature difference of {diff} degrees."

        else: # tz_calc
            tz1, tz2 = rng.sample(TIMEZONES[:6], 2)
            user_prompt = f"What is the time difference between {tz1} and {tz2}{q_mark}"
            thought1 = "First, I will query the current time in the first timezone."
            call1 = f"get_time(timezone=\"{tz1}\")"
            result1 = json.dumps({"time": "08:00", "utc_offset": -5})
            thought2 = "Next, I will query the current time in the second timezone."
            call2 = f"get_time(timezone=\"{tz2}\")"
            result2 = json.dumps({"time": "22:00", "utc_offset": 9})
            answer = f"{tz1} is UTC-5 and {tz2} is UTC+9, representing a 14-hour time difference."

        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought1}</THINK>"
            f"<CALL>{call1}</CALL>"
            f"<RESULT>{result1}</RESULT>"
            f"<THINK>{thought2}</THINK>"
            f"<CALL>{call2}</CALL>"
            f"<RESULT>{result2}</RESULT>"
            f"<ANSWER>{answer}</ANSWER>"
        )

    elif category == "nested_json_schema":
        # Complex tools with nested dictionaries, lists, and boolean flags
        schema_type = rng.choice(["db_query", "event_create", "file_manage", "api_fetch"])
        if schema_type == "db_query":
            dept = rng.choice(DEPARTMENTS)
            limit = rng.randint(3, 10)
            user_prompt = rng.choice([
                f"Query the top {limit} active records in the {dept} department{p_mark}",
                f"Query active records in the {dept} department with limit {limit}{p_mark}",
                f"Fetch the top {limit} active entries from {dept}{p_mark}",
            ])
            thought = "I will invoke database_query with a nested filter and options schema."
            call = f"database_query(filter={{\"department\": \"{dept}\", \"status\": \"active\"}}, options={{\"limit\": {limit}, \"sort\": \"desc\"}})"
            result = json.dumps({"status": "success", "count": limit, "records": [{"id": 100 + i, "dept": dept} for i in range(min(limit, 3))]})
            answer = f"Found {limit} active records in {dept} according to your filter specifications."

        elif schema_type == "event_create":
            attendees = ["alice@corp.internal", "bob@corp.internal"]
            user_prompt = f"Schedule a Sprint Sync meeting for tomorrow from 10:00 to 11:00 UTC with Alice and Bob{p_mark}"
            thought = "I need to call create_event with a nested event schedule and attendee list."
            call = "create_event(event={\"title\": \"Sprint Sync\", \"schedule\": {\"date\": \"2026-11-01\", \"start\": \"10:00\", \"end\": \"11:00\", \"tz\": \"UTC\"}}, attendees=[\"alice@corp.internal\", \"bob@corp.internal\"])"
            result = json.dumps({"event_id": "evt_98214", "status": "confirmed"})
            answer = "Successfully scheduled the Sprint Sync meeting with Alice and Bob for 10:00-11:00 UTC (Event ID: evt_98214)."

        elif schema_type == "file_manage":
            path = "/configs/model_t4.json"
            user_prompt = f"Save model configuration to {path} with fp16 enabled and 2 workers{p_mark}"
            thought = "Calling file_manager with action write and a nested file payload."
            call = f"file_manager(action=\"write\", file={{\"path\": \"{path}\", \"format\": \"json\"}}, content={{\"fp16\": true, \"workers\": 2}})"
            result = json.dumps({"status": "ok", "bytes_written": 48})
            answer = f"Configuration file has been written to {path} with fp16=True and 2 workers."

        else: # api_fetch
            endpoint = "/v2/metrics/latency"
            user_prompt = f"Fetch latency metrics for EU region from {endpoint}{p_mark}"
            thought = "Calling fetch_api with endpoint and nested filter parameters."
            call = f"fetch_api(endpoint=\"{endpoint}\", params={{\"region\": \"EU\", \"metric\": \"p99\"}}, method=\"GET\")"
            result = json.dumps({"status": 200, "data": {"p99_ms": 14.2}})
            answer = f"Retrieved p99 latency for EU region from {endpoint}: 14.2 ms."

        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought}</THINK>"
            f"<CALL>{call}</CALL>"
            f"<RESULT>{result}</RESULT>"
            f"<ANSWER>{answer}</ANSWER>"
        )

    elif category == "error_recovery":
        # Robust error reasoning & recovery paths
        err_type = rng.choice(["div_zero", "city_not_found", "api_503_fallback", "table_missing"])
        if err_type == "div_zero":
            num = rng.choice([100, 150, 250, 500])
            user_prompt = f"Calculate {num} / 0"
            thought1 = "The user requests an arithmetic calculation. I will invoke the calculator tool."
            call1 = f"calculator(expr=\"{num}/0\")"
            result1 = json.dumps({"error": "ZeroDivisionError: division by zero"})
            thought2 = "The calculator returned ZeroDivisionError. Division by zero is mathematically undefined. I will explain this clearly."
            answer = "Error: Division by zero is undefined in mathematics. Please specify a non-zero denominator."
            full_text = (
                f"<USER>{user_prompt}</USER>"
                f"<THINK>{thought1}</THINK>"
                f"<CALL>{call1}</CALL>"
                f"<RESULT>{result1}</RESULT>"
                f"<THINK>{thought2}</THINK>"
                f"<ANSWER>{answer}</ANSWER>"
            )

        elif err_type == "city_not_found":
            city = rng.choice(["Atlantis", "Narnia"])
            user_prompt = f"What is the weather in {city}{q_mark}"
            thought1 = "The user requests current weather information. I will query the get_weather tool."
            call1 = f"get_weather(city=\"{city}\")"
            result1 = json.dumps({"status": "error", "code": 404, "message": f"City '{city}' not found"})
            thought2 = "get_weather returned 404. The requested city is not recognized. I must inform the user."
            answer = f"I could not retrieve weather data for {city} because it is not recognized in our weather database."
            full_text = (
                f"<USER>{user_prompt}</USER>"
                f"<THINK>{thought1}</THINK>"
                f"<CALL>{call1}</CALL>"
                f"<RESULT>{result1}</RESULT>"
                f"<THINK>{thought2}</THINK>"
                f"<ANSWER>{answer}</ANSWER>"
            )

        elif err_type == "api_503_fallback":
            city = "Tokyo"
            user_prompt = f"Check the primary API forecast for {city}{p_mark}"
            thought1 = "Attempting primary weather API fetch."
            call1 = f"fetch_api(endpoint=\"/v1/weather\", params={{\"city\": \"{city}\"}}, method=\"GET\")"
            result1 = json.dumps({"status": "error", "code": 503, "message": "Service Unavailable"})
            thought2 = "Primary API failed with 503. Recovering via fallback tool get_weather."
            call2 = f"get_weather(city=\"{city}\")"
            result2 = json.dumps({"temp": 22, "condition": "Sunny"})
            answer = "Primary service was temporarily unavailable, but fallback weather confirmed Tokyo is Sunny at 22C."
            full_text = (
                f"<USER>{user_prompt}</USER>"
                f"<THINK>{thought1}</THINK>"
                f"<CALL>{call1}</CALL>"
                f"<RESULT>{result1}</RESULT>"
                f"<THINK>{thought2}</THINK>"
                f"<CALL>{call2}</CALL>"
                f"<RESULT>{result2}</RESULT>"
                f"<ANSWER>{answer}</ANSWER>"
            )

        else: # table_missing
            user_prompt = f"Query records from table 'legacy_metrics'{p_mark}"
            thought1 = "Querying database for the requested table."
            call1 = "database_query(filter={\"table\": \"legacy_metrics\"}, options={\"limit\": 5})"
            result1 = json.dumps({"status": "error", "code": "TABLE_NOT_FOUND", "message": "Table does not exist"})
            thought2 = "TABLE_NOT_FOUND error encountered. Informing user of invalid table name."
            answer = "The requested database table 'legacy_metrics' was not found. Please verify table name."
            full_text = (
                f"<USER>{user_prompt}</USER>"
                f"<THINK>{thought1}</THINK>"
                f"<CALL>{call1}</CALL>"
                f"<RESULT>{result1}</RESULT>"
                f"<THINK>{thought2}</THINK>"
                f"<ANSWER>{answer}</ANSWER>"
            )

    elif category == "multi_turn":
        # Multi-turn conversational dialogue with state continuity
        city = rng.choice(CITIES)
        temp = rng.randint(10, 25)
        w_q = rng.choice([
            f"How is the weather in {city}{q_mark}",
            f"What is the weather in {city}{q_mark}",
            f"What is the weather on {city}{q_mark}",
        ])
        full_text = (
            f"<USER>Hi there!</USER>"
            f"<THINK>Conversational greeting without tool.</THINK>"
            f"<ANSWER>Hello! How can I assist you today?</ANSWER>"
            f"<USER>{w_q}</USER>"
            f"<THINK>The user requests current weather information. I will query the get_weather tool.</THINK>"
            f"<CALL>get_weather(city=\"{city}\")</CALL>"
            f"<RESULT>{{\"temp\": {temp}, \"condition\": \"Clear\"}}</RESULT>"
            f"<ANSWER>{city} is currently Clear with a temperature of {temp} degrees Celsius.</ANSWER>"
            f"<USER>Thank you!</USER>"
            f"<THINK>User acknowledges. Polite closing response.</THINK>"
            f"<ANSWER>You are very welcome! Let me know if you need anything else.</ANSWER>"
        )

    else: # no_tool
        greeting = rng.choice([
            ("Hello! How are you?", "I am doing well, thank you! How may I assist you today?"),
            ("Hi there!", "Hello! What can I help you with today?"),
            ("Thank you for your assistance.", "You are welcome! Let me know if you have any more questions."),
            ("Thank you!", "You are very welcome! Let me know if you need anything else."),
            ("Good morning!", "Good morning! How may I help you?"),
            ("Who are you?", "I am ScaledGPT, an AI assistant with structured tool-calling and high-digit arithmetic capabilities."),
            ("Explain the concept of gradient descent.", "Gradient descent is an optimization algorithm that iteratively moves model weights in the direction of steepest loss decrease."),
        ])
        user_prompt, ans = greeting
        thought = "Conversational request that does not require any tool. I will answer directly."
        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought}</THINK>"
            f"<ANSWER>{ans}</ANSWER>"
        )

    return {
        "category": category,
        "full_text": full_text,
    }


def generate_tool_dataset(n_samples: int = 4000, seed: int = 1337) -> List[Dict[str, str]]:
    """Generates an expanded tool interaction dataset across all 6 categories."""
    rng = random.Random(seed)
    dataset: List[Dict[str, str]] = []
    for _ in range(n_samples):
        ex = generate_tool_example(rng)
        full_text = ex["full_text"]
        prompt = full_text.split("</USER>")[0] + "</USER>"
        dataset.append({
            "task_type": "tools",
            "prompt": prompt,
            "full_text": full_text,
            "category": ex["category"],
        })
    return dataset


# -----------------------------------------------------------------------------
# 2b. Curriculum V5: Broad-Distribution Copy Spans & Parameterized Schemas
# -----------------------------------------------------------------------------
# V4 copy spans came from one narrow phonotactic inventory, so real-world
# spellings outside it (doubled vowels, lowercase connectors, leading articles)
# drifted at copy time. V5 widens the span distribution, strips articles from
# arguments, parameterizes every nested schema, and keeps <THINK> short so
# nested calls close within a 140-token decode budget.

SPAN_ONSETS_EXT = NONCE_ONSETS + [
    "", "", "b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t", "w", "y",
    "ph", "sh", "kh", "zh", "ts", "ng", "nj", "sr", "vr", "hy", "mb", "nd", "rh", "x",
]
SPAN_NUCLEI_EXT = NONCE_NUCLEI + ["aa", "ii", "uu", "ae", "ia", "io", "ua", "ei", "au", "eo", "ie", "ya"]
SPAN_CODAS_EXT = NONCE_CODAS + ["", "", "", "n", "r", "s", "l", "ng", "rm", "nd", "st", "x", "ch", "lt", "ss"]
SPAN_CONNECTORS = ["of", "van", "de", "da", "del", "es", "la", "al", "bin", "von", "le", "and", "y", "el"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"

V5_TIMEZONES = TIMEZONES + ["MST", "EET", "BST", "SGT", "NZST", "PDT", "EDT", "WET", "KST", "MSK"]
V5_REGIONS = ["EU", "US", "APAC", "LATAM", "MEA", "UK", "JP", "IN", "BR", "CA"]
V5_METRICS = ["latency", "errors", "uptime", "cpu", "memory", "requests", "traffic", "bandwidth"]
V5_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "tomorrow", "next week"]
V5_FILE_EXTS = ["txt", "json", "csv", "md", "log", "py", "yaml", "cfg"]
V5_RECORD_STATUSES = ["active", "inactive", "pending", "archived"]

NO_TOOL_OPENERS = [
    "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening",
    "Greetings", "Hi there", "Hello there", "Hey there", "Howdy",
]
NO_TOOL_BODIES = [
    "how are you", "how are you today", "how are you doing", "how is it going",
    "how have you been", "what's up", "nice to meet you", "how is your day",
    "how is your day going", "how are things", "are you there", "how are you doing today",
]
NO_TOOL_STANDALONE = [
    ("thanks", "Thank you"), ("thanks", "Thanks a lot"), ("thanks", "Thanks for the help"),
    ("thanks", "Thank you so much"), ("thanks", "Much appreciated"), ("thanks", "Great, thanks"),
    ("bye", "Goodbye"), ("bye", "See you later"), ("bye", "Good night"), ("bye", "Bye for now"),
    ("identity", "Who are you"), ("identity", "What can you do"), ("identity", "What is your name"),
    ("identity", "Are you a robot"), ("ack", "Okay"), ("ack", "Nice work"), ("ack", "That was helpful"),
    ("ack", "Cool"), ("ack", "Sounds good"), ("ack", "Got it"),
]
NO_TOOL_REPLIES = {
    "greeting": [
        "Hello! I am doing well, thank you. How can I help you today?",
        "Hi! Everything is running smoothly. What can I do for you?",
        "Hello! How may I assist you today?",
    ],
    "thanks": [
        "You are welcome! Let me know if you need anything else.",
        "Happy to help! Feel free to ask another question.",
    ],
    "bye": ["Goodbye! Have a great day.", "See you soon! Take care."],
    "identity": [
        "I am CBT, a compact assistant that can call tools for weather, time, search, unit conversion, and arithmetic.",
    ],
    "ack": ["Great! Let me know if there is anything else I can do.", "Glad to hear it. What else can I help with?"],
}


def _span_word(rng: random.Random) -> str:
    """Samples one lowercase word from a mixture of phonotactic, nonce, random-letter, and alphanumeric shapes."""
    r = rng.random()
    if r < 0.55:
        w = ""
        while not 2 <= len(w) <= 14:
            n_syl = rng.choice([1, 2, 2, 2, 3, 3, 4])
            w = "".join(
                rng.choice(SPAN_ONSETS_EXT) + rng.choice(SPAN_NUCLEI_EXT)
                + (rng.choice(SPAN_CODAS_EXT) if rng.random() < 0.5 else "")
                for _ in range(n_syl)
            )
    elif r < 0.75:
        w, _, _ = generate_nonce_entity(rng, allow_multi_word=False)
        w = w.lower()
    elif r < 0.90:
        w = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 12)))
    else:
        stem = "".join(rng.choice(LETTERS) for _ in range(rng.randint(1, 6)))
        num = str(rng.randint(0, 999))
        w = rng.choice([stem + num, stem + "-" + num, num + stem, stem.upper() + num])
    return w if len(w) >= 2 else w + rng.choice(LETTERS)


def generate_copy_span(rng: random.Random, max_words: int = 4, title_prob: float = 0.5) -> str:
    """Samples a 1-4 word span with mixed casing and optional lowercase connectors ('X de Y' shape)."""
    n_words = rng.choices([1, 2, 3, 4][:max_words], weights=[0.50, 0.30, 0.15, 0.05][:max_words], k=1)[0]
    words = [_span_word(rng) for _ in range(n_words)]
    while len(" ".join(words)) > 36:
        words = [_span_word(rng) for _ in range(n_words)]
    casing = rng.random()
    if casing < title_prob:
        words = [w[0].upper() + w[1:] for w in words]
    elif casing < title_prob + 0.05:
        words = [w.upper() for w in words]
    elif casing < title_prob + 0.10:
        words[0] = words[0][0].upper() + words[0][1:]
    if n_words >= 2 and rng.random() < 0.25:
        words.insert(rng.randint(1, n_words - 1), rng.choice(SPAN_CONNECTORS))
    return " ".join(words)


def _v5_entity(rng: random.Random) -> Tuple[str, str]:
    """Returns (entity, snippet), mixing broad copy spans with case-perturbed known entities."""
    if rng.random() < 0.70:
        ent = generate_copy_span(rng)
        return ent, "a topic described in verified reference sources."
    item = rng.choice(SEARCH_ENTITIES)
    ent = item["entity"]
    r = rng.random()
    if r < 0.25:
        ent = ent.lower()
    elif r < 0.35:
        ent = ent.title()
    return ent, item["snippet"]


def _wrap(user_prompt: str, thought: str, call: str, result: str, answer: str) -> str:
    return (
        f"<USER>{user_prompt}</USER><THINK>{thought}</THINK><CALL>{call}</CALL>"
        f"<RESULT>{result}</RESULT><ANSWER>{answer}</ANSWER>"
    )


def generate_search_v5_example(rng: random.Random) -> Dict[str, str]:
    """Search delegation with article-stripped arguments ('Tell me about the X' -> query='X')."""
    ent, snippet = _v5_entity(rng)
    q = "?" if rng.random() < 0.5 else ""
    p = "." if rng.random() < 0.5 else ""
    article = "an" if ent[0].lower() in "aeiou" else "a"
    user_prompt = rng.choice([
        f"What is {ent}{q}", f"What is the {ent}{q}", f"What is {article} {ent}{q}", f"What are {ent}{q}",
        f"What are the {ent}{q}", f"Tell me about {ent}{p}", f"Tell me about the {ent}{p}",
        f"Tell me about {article} {ent}{p}", f"Define {ent}{p}", f"Define the term {ent}{p}",
        f"Explain {ent}{p}", f"Explain the {ent}{p}", f"Search for {ent}{p}", f"Look up {ent}{p}",
        f"Who is {ent}{q}", f"Who was {ent}{q}", f"Find information about {ent}{p}",
        f"Give me information on {ent}{p}", f"I want to learn about {ent}{p}",
    ])
    thought = rng.choice([
        "The user is asking a factual question. I will query the search tool.",
        "The user requests factual information. Querying search.",
        "The user is seeking knowledge on this topic. I will search the web.",
    ])
    full_text = _wrap(
        user_prompt, thought, f"search(query=\"{ent}\")",
        json.dumps({"snippet": snippet}), f"According to search results, {ent} is {snippet}",
    )
    return {"category": "search_v5", "full_text": full_text}


def generate_single_tool_v5_example(rng: random.Random) -> Dict[str, str]:
    """Weather / calculator / time / unit_convert with broad spans and verbatim calculator expressions."""
    q = "?" if rng.random() < 0.5 else ""
    p = "." if rng.random() < 0.5 else ""
    sub = rng.choices(["weather", "calculator", "time", "unit_convert"], weights=[0.35, 0.35, 0.15, 0.15], k=1)[0]

    if sub == "weather":
        city = generate_copy_span(rng, max_words=3, title_prob=0.85) if rng.random() < 0.55 else rng.choice(CITIES)
        temp, cond = rng.randint(-10, 40), rng.choice(CONDITIONS)
        user_prompt = rng.choice([
            f"What is the weather in {city}{q}", f"What is the weather on {city}{q}",
            f"What is the weather for {city}{q}", f"What's the weather in {city}{q}",
            f"What is the weather like in {city}{q}", f"How is the weather in {city}{q}",
            f"How is the weather on {city}{q}", f"Weather in {city}{p}", f"Weather on {city}{p}",
            f"Weather for {city}{p}", f"Check the weather in {city}{p}", f"Check the weather for {city}{p}",
            f"Tell me the weather in {city}{p}", f"Forecast for {city}{p}", f"{city} weather{p}",
            f"Is it raining in {city}{q}", f"Temperature in {city}{p}",
        ])
        thought = rng.choice([
            "The user requests current weather information. I will query the get_weather tool.",
            "The user is asking for weather conditions. Querying get_weather.",
        ])
        call = f"get_weather(city=\"{city}\")"
        result = json.dumps({"temp": temp, "condition": cond, "units": "celsius"})
        answer = f"The weather in {city} is currently {cond} with a temperature of {temp} degrees Celsius."

    elif sub == "calculator":
        op = rng.choice(["+", "-", "*", "/"])
        if op == "/":
            b = rng.randint(2, 99)
            res = rng.randint(2, 999)
            a = res * b
        else:
            a = rng.randint(1, 10 ** rng.randint(1, 5) - 1)
            b = rng.randint(1, 10 ** rng.randint(1, 4) - 1)
            res = a + b if op == "+" else (a - b if op == "-" else a * b)
        expr = rng.choice([f"{a} {op} {b}", f"{a}{op}{b}"])
        user_prompt = rng.choice([
            f"Calculate {expr}{p}", f"What is {expr}{q}", f"Compute {expr}{p}", f"Evaluate {expr}{p}",
            f"Solve {expr}{p}", f"What's {expr}{q}", f"How much is {expr}{q}",
        ])
        thought = rng.choice([
            "The user requests an arithmetic calculation. I will invoke the calculator tool.",
            "The user is asking to evaluate a mathematical expression. Calling calculator.",
        ])
        call = f"calculator(expr=\"{expr}\")"
        result = json.dumps({"result": res})
        answer = f"The evaluated result of {expr} is {res}."

    elif sub == "time":
        # 40% synthetic uppercase codes so timezone is copied from the prompt, not recalled from a list
        tz = rng.choice(V5_TIMEZONES) if rng.random() < 0.6 else "".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 5))).upper()
        time_str = f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
        user_prompt = rng.choice([
            f"What is the current time in {tz}{q}", f"What is the time in {tz}{q}", f"What time is it in {tz}{q}",
            f"Current time in {tz}{p}", f"Time in {tz}{p}", f"Tell me the time in {tz}{p}",
            f"Check the time for {tz}{p}",
        ])
        thought = "The user is asking for the current time. Querying get_time."
        call = f"get_time(timezone=\"{tz}\")"
        result = json.dumps({"time": time_str, "timezone": tz})
        answer = f"The current time in {tz} is {time_str}."

    else:
        val = rng.randint(1, 1000)
        from_u, to_u, factor = rng.choice([
            ("km", "miles", 0.621371), ("miles", "km", 1.60934), ("kg", "lbs", 2.20462), ("lbs", "kg", 0.453592),
        ])
        user_prompt = rng.choice([
            f"Convert {val} {from_u} to {to_u}{p}", f"How many {to_u} is {val} {from_u}{q}",
            f"How many {to_u} are in {val} {from_u}{q}", f"What is {val} {from_u} in {to_u}{q}",
            f"Change {val} {from_u} into {to_u}{p}",
        ])
        thought = "The user requests a unit conversion. I will invoke the unit_convert tool."
        call = f"unit_convert(value={val}, from_unit=\"{from_u}\", to_unit=\"{to_u}\")"
        result = json.dumps({"value": round(val * factor, 2), "unit": to_u})
        answer = f"{val} {from_u} is equal to {round(val * factor, 2)} {to_u}."

    return {"category": f"single_{sub}", "full_text": _wrap(user_prompt, thought, call, result, answer)}


def generate_nested_v5_example(rng: random.Random) -> Dict[str, str]:
    """Parameterized compact nested schemas; short <THINK> keeps the full call under the decode budget."""
    p = "." if rng.random() < 0.5 else ""
    schema = rng.choice(["db_query", "event_create", "file_manage", "api_fetch"])

    if schema == "db_query":
        dept = rng.choice(DEPARTMENTS) if rng.random() < 0.6 else _span_word(rng).lower()[:12]
        n = rng.randint(1, 20)
        status = rng.choice(V5_RECORD_STATUSES)
        user_prompt = rng.choice([
            f"Query the top {n} {status} records in the {dept} department{p}",
            f"Query top {n} {status} records in {dept}{p}",
            f"Get {n} {status} records from {dept}{p}",
            f"Show the top {n} {status} {dept} records{p}",
            f"Fetch the top {n} {status} entries from {dept}{p}",
            f"List {n} {status} records in the {dept} team{p}",
        ])
        thought = "Calling database_query."
        call = f"database_query(filter={{\"department\": \"{dept}\", \"status\": \"{status}\"}}, options={{\"limit\": {n}}})"
        result = json.dumps({"status": "success", "count": n})
        answer = f"Found {n} {status} records in {dept}."

    elif schema == "event_create":
        title = generate_copy_span(rng, max_words=3, title_prob=0.9)
        h = rng.randint(0, 22)
        start = f"{h:02d}:{rng.choice([0, 15, 30, 45]):02d}"
        end = f"{h + 1:02d}:{start[-2:]}"
        mode = rng.random()
        if mode < 0.3:
            day = rng.choice(V5_DAYS)
            on = "" if day in ("tomorrow", "next week") else "on "
            user_prompt = rng.choice([
                f"Create a calendar event for {title} {on}{day}{p}",
                f"Schedule {title} {on}{day}{p}",
                f"Add {title} to my calendar {on}{day}{p}",
                f"Book {title} {on}{day}{p}",
            ])
            call = f"create_event(event={{\"title\": \"{title}\", \"date\": \"{day}\"}})"
            start = day
        elif mode < 0.55:
            user_prompt = rng.choice([
                f"Schedule {title} from {start} to {end}{p}",
                f"Create a calendar event for {title} from {start} to {end}{p}",
                f"Book {title} between {start} and {end}{p}",
            ])
            call = f"create_event(event={{\"title\": \"{title}\", \"start\": \"{start}\", \"end\": \"{end}\"}})"
        else:
            user_prompt = rng.choice([
                f"Create a calendar event for {title} at {start}{p}",
                f"Create an event called {title} at {start}{p}",
                f"Schedule {title} at {start}{p}",
                f"Add {title} to my calendar at {start}{p}",
                f"Set up {title} at {start} tomorrow{p}",
            ])
            call = f"create_event(event={{\"title\": \"{title}\", \"start\": \"{start}\"}})"
        thought = "Calling create_event."
        result = json.dumps({"event_id": f"evt_{rng.randint(10000, 99999)}", "status": "confirmed"})
        answer = f"Scheduled {title} for {start}."

    elif schema == "file_manage":
        fname = f"{_span_word(rng).lower()[:10]}.{rng.choice(V5_FILE_EXTS)}"
        if rng.random() < 0.3:
            fname = f"{rng.choice(['docs', 'data', 'logs', 'tmp', 'notes'])}/{fname}"
        action = rng.choices(["write", "read", "delete"], weights=[0.7, 0.15, 0.15], k=1)[0]
        if action == "write":
            content = generate_copy_span(rng, max_words=3, title_prob=0.6)
            user_prompt = rng.choice([
                f"Save {fname} with content {content}{p}",
                f"Write {content} to {fname}{p}",
                f"Create file {fname} containing {content}{p}",
                f"Save the text {content} in {fname}{p}",
            ])
            call = f"file_manager(action=\"write\", file={{\"path\": \"{fname}\"}}, content=\"{content}\")"
        else:
            verb = "Read" if action == "read" else "Delete"
            user_prompt = rng.choice([f"{verb} {fname}{p}", f"{verb} the file {fname}{p}"])
            call = f"file_manager(action=\"{action}\", file={{\"path\": \"{fname}\"}})"
        thought = "Calling file_manager."
        result = json.dumps({"status": "ok"})
        answer = f"File operation {action} completed for {fname}."

    else:
        segs = [_span_word(rng).lower()[:8] for _ in range(rng.randint(1, 2))]
        endpoint = f"/v{rng.randint(1, 3)}/" + "/".join(segs)
        region, metric = rng.choice(V5_REGIONS), rng.choice(V5_METRICS)
        user_prompt = rng.choice([
            f"Fetch {metric} metrics for {region} from {endpoint}{p}",
            f"Fetch {metric} metrics for {region} region from {endpoint}{p}",
            f"Get {metric} data from {endpoint} for {region}{p}",
            f"Call {endpoint} for {region} {metric}{p}",
        ])
        thought = "Calling fetch_api."
        call = f"fetch_api(endpoint=\"{endpoint}\", params={{\"region\": \"{region}\", \"metric\": \"{metric}\"}})"
        result = json.dumps({"status": 200, "data": {"value": round(rng.random() * 100, 1)}})
        answer = f"Retrieved {metric} for {region} from {endpoint}."

    return {"category": f"nested_{schema}", "full_text": _wrap(user_prompt, thought, call, result, answer)}


def generate_error_v5_example(rng: random.Random) -> Dict[str, str]:
    """Error recovery traces using the V5 compact schemas."""
    err = rng.choice(["div_zero", "city_not_found", "api_503_fallback", "table_missing"])
    if err == "div_zero":
        num = rng.randint(1, 9999)
        user_prompt = f"Calculate {num} / 0"
        steps = [
            ("The user requests an arithmetic calculation. I will invoke the calculator tool.",
             f"calculator(expr=\"{num} / 0\")", json.dumps({"error": "ZeroDivisionError: division by zero"})),
        ]
        final_thought = "The calculator returned ZeroDivisionError. I will explain this clearly."
        answer = "Division by zero is undefined. Please provide a non-zero denominator."
    elif err == "city_not_found":
        city = generate_copy_span(rng, max_words=2, title_prob=1.0)
        user_prompt = f"What is the weather in {city}"
        steps = [
            ("The user requests current weather information. I will query the get_weather tool.",
             f"get_weather(city=\"{city}\")", json.dumps({"status": "error", "code": 404})),
        ]
        final_thought = "get_weather returned 404. I must inform the user."
        answer = f"I could not retrieve weather data for {city} because it is not recognized."
    elif err == "api_503_fallback":
        city = rng.choice(CITIES)
        user_prompt = f"Check the primary API forecast for {city}"
        steps = [
            ("Calling fetch_api.", f"fetch_api(endpoint=\"/v1/weather\", params={{\"city\": \"{city}\"}})",
             json.dumps({"status": 503})),
            ("Primary API failed with 503. Falling back to get_weather.", f"get_weather(city=\"{city}\")",
             json.dumps({"temp": 22, "condition": "Sunny"})),
        ]
        final_thought = ""
        answer = f"The primary service was unavailable, but the fallback reports {city} is Sunny at 22C."
    else:
        table = _span_word(rng).lower()[:12]
        user_prompt = f"Query records from table {table}"
        steps = [
            ("Calling database_query.", f"database_query(filter={{\"table\": \"{table}\"}}, options={{\"limit\": 5}})",
             json.dumps({"status": "error", "code": "TABLE_NOT_FOUND"})),
        ]
        final_thought = "TABLE_NOT_FOUND. Informing the user."
        answer = f"The table {table} was not found. Please verify the table name."

    text = f"<USER>{user_prompt}</USER>"
    for thought, call, result in steps:
        text += f"<THINK>{thought}</THINK><CALL>{call}</CALL><RESULT>{result}</RESULT>"
    if final_thought:
        text += f"<THINK>{final_thought}</THINK>"
    text += f"<ANSWER>{answer}</ANSWER>"
    return {"category": "error_v5", "full_text": text}


def generate_no_tool_v5_example(rng: random.Random) -> Dict[str, str]:
    """Compositional chit-chat (greetings with 'today'/'doing' etc.) routed directly to <ANSWER>."""
    if rng.random() < 0.6:
        opener, body = rng.choice(NO_TOOL_OPENERS), rng.choice(NO_TOOL_BODIES)
        sep = rng.choice([", ", "! ", " ", ". "])
        user_prompt = f"{opener}{sep}{body}{rng.choice(['?', '', '!'])}"
        if rng.random() < 0.15:
            user_prompt = user_prompt.lower()
        kind = "greeting"
    else:
        kind, user_prompt = rng.choice(NO_TOOL_STANDALONE)
        user_prompt += rng.choice(["", ".", "!", "?"] if kind == "identity" else ["", ".", "!"])
    thought = "Conversational request that does not require any tool. I will answer directly."
    full_text = f"<USER>{user_prompt}</USER><THINK>{thought}</THINK><ANSWER>{rng.choice(NO_TOOL_REPLIES[kind])}</ANSWER>"
    return {"category": "no_tool_v5", "full_text": full_text}


def generate_v5_tool_example(rng: random.Random) -> Dict[str, str]:
    """Samples one non-search V5 tool trace; chains and multi-turn reuse the V4 generator."""
    category = rng.choices(
        ["single", "nested", "no_tool", "error", "chain", "multi_turn"],
        weights=[0.42, 0.24, 0.14, 0.08, 0.06, 0.06],
        k=1,
    )[0]
    if category == "single":
        return generate_single_tool_v5_example(rng)
    if category == "nested":
        return generate_nested_v5_example(rng)
    if category == "no_tool":
        return generate_no_tool_v5_example(rng)
    if category == "error":
        return generate_error_v5_example(rng)
    return generate_tool_example(rng, category="multi_tool_chain" if category == "chain" else "multi_turn")


def to_tool_item(example: Dict[str, str]) -> Dict[str, str]:
    """Wraps a generated trace into the encode_joint_batch item format."""
    full_text = example["full_text"]
    return {
        "task_type": "tools",
        "prompt": full_text.split("</USER>")[0] + "</USER>",
        "full_text": full_text,
        "category": example["category"],
    }


def generate_joint_dataset(
    n_math: int = 4000,
    n_tools: int = 4000,
    seed: int = 42,
    max_train_digits: int = 15,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Generates dual datasets for math curriculum and expanded tools."""
    math_samples = generate_arithmetic_equations(n_samples=n_math, seed=seed, max_train_digits=max_train_digits)
    tool_samples = generate_tool_dataset(n_samples=n_tools, seed=seed + 100)
    return math_samples, tool_samples


# -----------------------------------------------------------------------------
# 3. Batch Encoder & Strict Loss Masker
# -----------------------------------------------------------------------------

def encode_joint_batch(
    math_items: List[Dict[str, str]],
    tool_items: List[Dict[str, str]],
    tokenizer: ScaledTokenizer,
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes balanced joint batch of math and tool sequences.
    Strict Prompt & Environment Loss Masking:
    - Math: masks everything up to and including '<THINK>' (loss on scratchpad + answer).
    - Tools:
      - Masks everything inside '<USER>...</USER>' except at '</USER>' (where target is '<THINK>').
      - Masks everything inside '<RESULT>...</RESULT>' except at '</RESULT>' (where target is '<THINK>' or '<ANSWER>').
      - Supervised: reasoning traces '<THINK>', tool calls '<CALL>', and answers '<ANSWER>'.
    """
    items = math_items + tool_items
    B = len(items)

    encoded_seqs = [tokenizer.encode(item["full_text"], add_bos=True, add_eos=True) for item in items]
    calc_max_len = max(len(s) for s in encoded_seqs) - 1
    if max_len is not None:
        batch_max = min(calc_max_len, max_len)
    else:
        batch_max = calc_max_len

    x_tensor = torch.full((B, batch_max), tokenizer.pad_id, dtype=torch.long)
    y_tensor = torch.full((B, batch_max), -100, dtype=torch.long)

    think_s_id = tokenizer.think_s_id
    user_s_id = tokenizer.user_s_id
    user_e_id = tokenizer.user_e_id
    result_s_id = tokenizer.result_s_id
    result_e_id = tokenizer.result_e_id

    for i, (item, tokens) in enumerate(zip(items, encoded_seqs)):
        # Truncate if exceeding batch_max + 1
        if len(tokens) > batch_max + 1:
            tokens = tokens[: batch_max + 1]

        n = len(tokens) - 1
        t_tok = torch.tensor(tokens, dtype=torch.long)
        x_tensor[i, :n] = t_tok[:-1]
        y_tensor[i, :n] = t_tok[1:].clone()

        if item["task_type"] == "math":
            # Mask prompt up to '<THINK>'
            think_matches = (t_tok[:-1] == think_s_id).nonzero(as_tuple=True)[0]
            if len(think_matches) > 0:
                mask_until = think_matches[0].item()
                y_tensor[i, :mask_until] = -100
        else:
            # Segment-aware masking for single-turn, multi-turn, and tool responses
            mask = torch.zeros(n, dtype=torch.bool)
            in_user = False
            in_result = False

            for pos in range(n):
                tok_id = t_tok[pos].item()
                if tok_id == user_s_id:
                    in_user = True
                elif tok_id == result_s_id:
                    in_result = True

                if in_user:
                    if tok_id == user_e_id:
                        in_user = False
                        mask[pos] = False
                    else:
                        mask[pos] = True
                elif in_result:
                    if tok_id == result_e_id:
                        in_result = False
                        mask[pos] = False
                    else:
                        mask[pos] = True
                elif pos == 0:
                    # BOS token preceding <USER>
                    mask[pos] = True

            y_tensor[i, :n][mask] = -100

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor
