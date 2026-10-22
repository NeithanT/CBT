import random
import json
from typing import List, Dict, Tuple, Any


CITIES = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Madrid", "Rome", "Seoul"]
CONDITIONS = ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy", "Foggy"]
TIMEZONES = ["UTC", "EST", "PST", "GMT", "JST", "CET", "AEST"]
SEARCH_TOPICS = [
    ("Python creator", "Guido van Rossum"),
    ("capital of Australia", "Canberra"),
    ("speed of light", "299,792 km/s"),
    ("tallest mountain on Earth", "Mount Everest"),
    ("deepest ocean trench", "Mariana Trench"),
    ("inventor of the telephone", "Alexander Graham Bell"),
    ("currency of Japan", "Yen"),
    ("largest planet in solar system", "Jupiter"),
    ("author of Hamlet", "William Shakespeare"),
    ("chemical symbol for Gold", "Au"),
]
UNIT_PAIRS = [
    ("km", "miles", 0.621371),
    ("miles", "km", 1.60934),
    ("kg", "lbs", 2.20462),
    ("lbs", "kg", 0.453592),
    ("meters", "feet", 3.28084),
]


def generate_tool_example(rng: random.Random) -> Dict[str, str]:
    """Generates a structured synthetic tool calling example."""
    category = rng.choice(["calculator", "weather", "search", "time", "unit_convert", "no_tool"])

    if category == "calculator":
        op = rng.choice(["+", "-", "*", "/"])
        if op == "+":
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            res = a + b
        elif op == "-":
            a, b = rng.randint(50, 999), rng.randint(10, 49)
            res = a - b
        elif op == "*":
            a, b = rng.randint(5, 50), rng.randint(2, 20)
            res = a * b
        else:
            b = rng.randint(2, 10)
            res = rng.randint(5, 50)
            a = res * b

        expr = f"{a}{op}{b}"
        user_prompt = rng.choice([
            f"What is {expr}?",
            f"Calculate {expr}",
            f"Compute the value of {expr}.",
            f"Can you solve {expr} for me?",
        ])
        thought = f"The user needs calculation for {expr}. I will call the calculator tool."
        call = f"calculator(expr=\"{expr}\")"
        result = json.dumps({"result": res})
        answer = f"The result of {expr} is {res}."

    elif category == "weather":
        city = rng.choice(CITIES)
        temp = rng.randint(-5, 35)
        cond = rng.choice(CONDITIONS)
        user_prompt = rng.choice([
            f"What is the weather in {city}?",
            f"How is the weather in {city} right now?",
            f"Can you check the forecast for {city}?",
            f"Weather update for {city} please.",
        ])
        thought = f"The user is asking about the weather in {city}. I should call get_weather."
        call = f"get_weather(city=\"{city}\")"
        result = json.dumps({"temp": temp, "condition": cond})
        answer = f"The weather in {city} is currently {cond} with a temperature of {temp} degrees Celsius."

    elif category == "search":
        topic, fact = rng.choice(SEARCH_TOPICS)
        user_prompt = rng.choice([
            f"Who or what is {topic}?",
            f"Tell me about {topic}.",
            f"Search information on {topic}.",
            f"What is the {topic}?",
        ])
        thought = f"The user wants factual information about {topic}. I will search the web."
        call = f"search(query=\"{topic}\")"
        result = json.dumps({"snippet": fact})
        answer = f"According to search results, {topic} is {fact}."

    elif category == "time":
        tz = rng.choice(TIMEZONES)
        hr = rng.randint(0, 23)
        mn = rng.randint(0, 59)
        time_str = f"{hr:02d}:{mn:02d}"
        user_prompt = rng.choice([
            f"What time is it in {tz}?",
            f"Current time in {tz} timezone?",
            f"Check the clock in {tz}.",
        ])
        thought = f"The user requests the current time in {tz}. I will query get_time."
        call = f"get_time(timezone=\"{tz}\")"
        result = json.dumps({"time": time_str})
        answer = f"The current time in {tz} is {time_str}."

    elif category == "unit_convert":
        from_u, to_u, factor = rng.choice(UNIT_PAIRS)
        val = rng.randint(2, 50)
        converted = round(val * factor, 2)
        user_prompt = rng.choice([
            f"Convert {val} {from_u} to {to_u}.",
            f"How many {to_u} is {val} {from_u}?",
            f"{val} {from_u} in {to_u}",
        ])
        thought = f"The user wants unit conversion from {from_u} to {to_u}. Calling unit_convert."
        call = f"unit_convert(value={val}, from_unit=\"{from_u}\", to_unit=\"{to_u}\")"
        result = json.dumps({"value": converted, "unit": to_u})
        answer = f"{val} {from_u} is equal to {converted} {to_u}."

    else: # no_tool
        greeting = rng.choice([
            ("Hello! How are you?", "I am doing well, thank you! How can I assist you today?"),
            ("Hi there!", "Hello! What can I help you with today?"),
            ("Thank you for your help.", "You are welcome! Let me know if you need anything else."),
            ("Good morning!", "Good morning! How may I help you?"),
            ("Who are you?", "I am an AI assistant equipped with tool-calling capabilities."),
        ])
        user_prompt, ans = greeting
        thought = "This is a conversational greeting that does not require any tool. I will answer directly."
        call = "none"
        result = ""
        answer = ans

    # Full structured text format
    if call != "none":
        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought}</THINK>"
            f"<CALL>{call}</CALL>"
            f"<RESULT>{result}</RESULT>"
            f"<ANSWER>{answer}</ANSWER>"
        )
    else:
        full_text = (
            f"<USER>{user_prompt}</USER>"
            f"<THINK>{thought}</THINK>"
            f"<ANSWER>{answer}</ANSWER>"
        )

    return {
        "category": category,
        "user_prompt": user_prompt,
        "thought": thought,
        "call": call,
        "result": result,
        "answer": answer,
        "full_text": full_text,
    }


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> List[Dict[str, str]]:
    """Generates a dataset of diverse tool calling interactions."""
    rng = random.Random(seed)
    return [generate_tool_example(rng) for _ in range(n_samples)]
