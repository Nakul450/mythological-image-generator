import os
import time
import json
import requests
from bs4 import BeautifulSoup

# Create dataset folders
image_folder = "dataset/images"
os.makedirs(image_folder, exist_ok=True)

# Set search queries
queries = [
    "Krishna giving Bhagavad Gita sermon",
    "Ravana abducting Sita",
    "Hanuman lifting mountain",
    "Arjuna shooting arrows",
    "Battle of Kurukshetra",
    "Shiva as Neelkanth",
    "Rama breaking the bow",
    "Draupadi's marriage",
    "Bheema defeating Duryodhana",
    "Karna's vow to fight Arjuna",
    "Lord Vishnu's incarnation as Vamana",
    "Sita's trial by fire",
    "The Pandavas in exile",
    "The Kurukshetra war chariot",
    "Krishna and Arjuna at Kurukshetra",
    "Yudhishthira gambling with Shakuni",
    "Ghatotkacha fighting in Kurukshetra",
    "The birth of the Pandavas",
    "Ravana in Lanka",
    "Indrajit firing the Brahmastra",
    "Sage Vyasa dictating Mahabharata",
    "Rama and Lakshmana searching for Sita",
    "Hanuman burning Lanka",
    "The great battle of Ramayana",
    "Bhishma's vow of celibacy",
    "Kunti summoning the sun god",
    "Lord Ram's coronation",
    "Duryodhana and Bhima wrestling",
    "Sage Valmiki writing the Ramayana",
    "Lord Vishnu's ten avatars",
    "Narada's visit to Lord Vishnu",
    "The Pandavas' secret weapon",
    "The battle between Ram and Ravan",
    "Shurpanakha's humiliation",
    "Lord Ram crossing the ocean to Lanka",
    "Kumbhakarna's awakening",
    "The birth of Ravana",
    "Lord Krishna's childhood",
    "The fall of Lanka",
    "Vasudeva carrying Krishna across the Yamuna",
    "Rama meeting his army",
    "Krishna lifting the mountain",
    "Karna and Arjuna's rivalry",
    "The defeat of Dushasana",
    "Sage Agastya's visit to Ram",
    "Drona teaching the Pandavas",
    "The meeting of Sita and Ravana",
    "Arjuna receiving the Pashupatastra",
    "Lord Vishnu resting on Ananta",
    "The birth of Lakshmana",
    "Lord Shiva and Parvati in the Himalayas",
    "Ravana's ten heads",
    "The defeat of Ravana",
    "Karna's charity",
    "Sita's abduction",
    "The battle of the five brothers",
    "Vishwamitra's training of Ram and Lakshmana",
    "The Sage Vasishta",
    "Arjuna in meditation",
    "Hanuman crossing the ocean",
    "The death of Bhishma",
    "Rama's exile into the forest",
    "The meeting of Lord Rama and Hanuman",
    "Ganga descending to the Earth",
    "Rama preparing for the war",
    "Arjuna's divine weapons",
    "The killing of Shishupala",
    "Ravana's defeat in battle",
    "The story of Prahlada",
    "The birth of Lord Krishna",
    "Sage Bharadwaja's visit to Ram",
    "Rama with his bow",
    "The birth of Balarama",
    "Arjuna receiving the divine chariot",
    "The story of Prahlada's devotion",
    "Rama and the golden deer",
    "The Rakshasas' army",
    "The Pandavas' escape from the palace",
    "The battle between Vali and Sugriva",
    "Lord Shiva and Ganga",
    "Vishnu's incarnation as Narasimha",
    "The final battle between Ram and Ravana",
    "Sage Agastya's teachings",
    "The story of Ahalya",
    "The victory of Rama over Ravana",
    "Krishna stealing the butter",
    "The churning of the ocean",
    "Rama's encounter with the demoness Surpanakha",
    "The birth of Kauravas",
    "Krishna's childhood pranks",
    "The curse of the Pandavas",
    "The story of King Harishchandra",
    "The story of Prahlada and Holika",
    "The story of Rishi Markandeya",
    "The sage Narada's stories",
    "Rama returning to Ayodhya",
    "Lord Vishnu's consort Lakshmi",
    "The Pandavas' rule in Indraprastha",
    "The killing of Mahishasura",
    "The story of Shabari",
    "Rama and the golden deer",
    "The exile of Sita",
    "Bheema killing Duryodhana",
    "Krishna and the cowherds",
    "Lord Vishnu's return to Vaikuntha",
    "The Rajasuya Yajna of Yudhishthira",
    "Arjuna's penance for the Pasupata Astra",
    "The story of Ganga and Bhishma",
    "Lord Shiva and Parvati",
    "The creation of the universe",
    "The story of Narada and the devas",
    "Karna and his golden armor",
    "The story of Kunti's vow",
    "Lord Krishna as the charioteer",
    "The testing of Sita",
    "The battle of Lanka",
    "The death of Ravana's son",
    "The curse of Shakuni",
    "The story of Jatayu's death"

]


def fetch_image_urls(query, num_images=5):
    """Fetches image URLs from Google Image Search."""
    search_url = f"https://www.google.com/search?tbm=isch&q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    images = [img["src"] for img in soup.find_all("img") if "src" in img.attrs]
    return images[:num_images]


def download_image(img_url, file_path):
    """Downloads and saves an image from a URL."""
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
    return False


captions = []

for query in queries:
    print(f"🔍 Searching images for: {query}")
    image_urls = fetch_image_urls(query)

    for idx, img_url in enumerate(image_urls):
        file_name = f"{image_folder}/{query.replace(' ', '_')}_{idx}.jpg"
        if download_image(img_url, file_name):
            captions.append({"file": file_name, "caption": query})

# Save captions as JSON
with open("dataset/captions.json", "w") as f:
    json.dump(captions, f, indent=4)

print("✅ Dataset collection complete! Images & captions saved.")
