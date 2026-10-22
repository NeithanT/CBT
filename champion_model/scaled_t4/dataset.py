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

from champion_model.scaled_t4.tokenizer import ScaledTokenizer


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
]
CONDITIONS = ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy", "Foggy", "Clear"]
TIMEZONES = ["UTC", "EST", "PST", "GMT", "JST", "CET", "AEST", "CST", "IST", "HST"]

# Nonce word phonetic generators to force pure Attention Induction Copying
NONCE_ONSETS = ["bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "qu", "v", "z", "k", "j"]
NONCE_NUCLEI = ["a", "e", "i", "o", "u", "y", "ai", "ea", "ee", "oa", "oo", "ou"]
NONCE_CODAS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "z", "ck", "nk", "mp", "sh", "th", "nt", "rt"]


def generate_nonce_entity(rng: random.Random) -> Tuple[str, str, str]:
    """Generates an arbitrary pseudo-word entity (singular, plural, type)."""
    n_syl = rng.choice([1, 2, 2, 3])
    parts = []
    for _ in range(n_syl):
        parts.append(rng.choice(NONCE_ONSETS) + rng.choice(NONCE_NUCLEI) + rng.choice(NONCE_CODAS))
    singular = "".join(parts)
    plural = singular + ("es" if singular[-1] in "sxz" else "s")
    ent_type = rng.choice(["concept", "plant", "creature", "artifact", "substance"])
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

def generate_tool_example(rng: random.Random) -> Dict[str, str]:
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
    """
    category = rng.choices(
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
                # Nonce city to force pure copy induction
                city = generate_nonce_entity(rng)[0].capitalize()
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
                f"Weather in {city}{p_mark}",
                f"Weather on {city}{p_mark}",
                f"Weather for {city}{p_mark}",
                f"Check the weather in {city}{p_mark}",
                f"Check the weather for {city}{p_mark}",
                f"Tell me the weather in {city}{p_mark}",
                f"Forecast for {city}{p_mark}",
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
                ent, plural, ent_type = generate_nonce_entity(rng)
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
