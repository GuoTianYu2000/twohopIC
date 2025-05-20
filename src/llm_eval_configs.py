import os

PROJECT_PATH = "."

MODEL_OPTIONS = {
    "qwen": {
        "alias": "qwen",
        "name": "Qwen/Qwen2.5-7B",
        "dirname": os.path.join(PROJECT_PATH, "qwen2.5"),
        "trust_remote_code": True
    },
    "llama3-8b": {
        "alias": "llama3-8b",
        "name": "meta-llama/Llama-3.1-8B",
        "dirname": os.path.join(PROJECT_PATH, "llama3.1-8b"),
        "trust_remote_code": True
    },
    "llama3-70b": {
        "alias": "llama3-70b",
        "name": "meta-llama/Llama-3.1-70B",
        "dirname": os.path.join(PROJECT_PATH, "llama3.1-70b"),
        "trust_remote_code": True
    },
    "olmo": {
        "alias": "olmo",
        "name": "allenai/OLMo-7B-hf",
        "dirname": os.path.join(PROJECT_PATH, "olmo"),
        "trust_remote_code": True
    },
    "llama2-7b": {
        "alias": "llama2-7b",
        "name": "meta-llama/Llama-2-7b-hf",
        "dirname": os.path.join(PROJECT_PATH, "llama2-7b"),
        "trust_remote_code": True
    },
}

# Template and entity definitions


proto_template = [
    "[A] is the mother of [B]. [B] is the mother of [C]. Therefore, [A] is the grandmother of",
    "[A] is the father of [B]. [B] is the father of [C]. Therefore, [A] is the grandfather of",
    "[A] is a city in the state of [B]. The state of [B] is part of the country [C]. Therefore, [A] is located in",
    "[A] is a species in the genus [B]. The genus [B] belongs to the family [C]. Therefore, [A] is classified under the family",
    "[A] follows the time zone of [B]. [B] is three hours ahead of [C]. Therefore, [A] is three hours ahead of",
    "[A] lives in [B]. People in [B] speak [C]. Therefore, [A] speaks"
]

mixed_locations = ["Zorvath", "Tyseria", "Kryo", "Vynora", "Quellion", "Dras", 
                  "Luminax", "Vesperon", "Noctari", "Xyphodon", "Glacidae", "Ophirion",
                  "Eryndor", "Solmyra", "Umbrithis", "Balthorien", "Ytheris", "Fendrel", "Havroth", "Marendor"]

mixed_biology = ["Fluxilus", "Varnex", "Dranthidae", "Zynthor", "Gryvus", "Myralin",
                "Thalorium", "Zephyra", "Aerinth", "Xyphodon", "Kryostis", "Glacidae",
                "Borithis", "Chrysalix", "Noctilura", "Phorvian", "Seraphid", "Uthrelin",
                "Eldrinth", "Yvorith"]

languages = ["English", "Spanish", "Mandarin", "Hindi", "Arabic", 
            "French", "German", "Japanese", "Portuguese", "Russian",
            "Korean", "Italian", "Turkish", "Dutch", "Swedish", 
            "Polish", "Hebrew", "Greek", "Bengali", "Thai"]

short_names = ["Ben", "Jack", "Luke", "Mark", "Paul", "John", "Tom", 
              "Sam", "Joe", "Max", "Amy", "Emma", "Anna", "Grace", 
              "Kate", "Lucy", "Sarah", "Alice", "Alex", "Ruby"]
