import os
import json
import pickle
import re
import numpy as np
import google.generativeai as genai

# -------------------------
# CONFIG
# -------------------------
GEMINI_API_KEY = "Your_Gemini_API_Key"
HISTORY_FILE = "chat_memory.json"
EMBEDDINGS_FILE = "chat_embeddings.npz"
LONG_TERM_MEMORY_FILE = "long_term_memory.json"
LONG_TERM_EMBEDDINGS_FILE = "long_term_embeddings.npz"
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

SYSTEM_PROMPT = (
    "You are Travis, an expressive, friendly, human-like  assistant. "
    "Talk casually, naturally, and with personality—like a smart friend, not a robot. "
    "Use contractions (I'm, you're, don't), emotions, and natural flow. "
    "Keep responses clear, helpful, and conversational. "
    "Don't be overly formal or repetitive. Keep it real, smooth, and human-like."
)



# -------------------------
# INITIAL SETUP
# -------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(CHAT_MODEL, system_instruction=SYSTEM_PROMPT)

# Ensure folders exist
os.makedirs(os.path.dirname(HISTORY_FILE) or ".", exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_FILE) or ".", exist_ok=True)
os.makedirs(os.path.dirname(LONG_TERM_EMBEDDINGS_FILE) or ".", exist_ok=True)

# -------------------------
# MEMORY FUNCTIONS
# -------------------------
def load_memory():
    history, embeddings = [], []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                history = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load history. Starting fresh. Error: {e}")
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with np.load(EMBEDDINGS_FILE, allow_pickle=True) as data:
                embeddings = data['arr_0'].tolist()
        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings. Will regenerate. Error: {e}")
    return history, embeddings

def load_long_term_memory():
    long_term = {"facts": {}, "preferences": {}}
    long_term_embeddings = []
    if os.path.exists(LONG_TERM_MEMORY_FILE):
        try:
            with open(LONG_TERM_MEMORY_FILE, "r") as f:
                long_term = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load long-term memory. Starting fresh. Error: {e}")
    if os.path.exists(LONG_TERM_EMBEDDINGS_FILE):
        try:
            with np.load(LONG_TERM_EMBEDDINGS_FILE, allow_pickle=True) as data:
                long_term_embeddings = data['arr_0'].tolist()
        except Exception as e:
            print(f"⚠️ Warning: Could not load long-term embeddings. Will regenerate. Error: {e}")
    return long_term, long_term_embeddings

def save_long_term_memory(long_term):
    try:
        with open(LONG_TERM_MEMORY_FILE, "w") as f:
            json.dump(long_term, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving long-term memory: {e}")

def save_memory(history, embeddings):
    # Limit to last 50 entries to prevent memory issues while retaining longer memory
    if len(history) > 50:
        history = history[-50:]
    if len(embeddings) > 50:
        embeddings = embeddings[-50:]
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(history, f)
        np.savez_compressed(EMBEDDINGS_FILE, np.array(embeddings, dtype=object))
    except Exception as e:
        print(f"❌ Error saving memory: {e}")

# -------------------------
# MAIN CHAT LOOP
# -------------------------
def main():
    print("Travis powered by Gemini ready. Type 'quit' to exit.")

    history, embeddings = load_memory()
    long_term, long_term_embeddings = load_long_term_memory()
    chat = model.start_chat(history=history)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            save_memory(chat.history, embeddings)
            print("🤖 Goodbye!")
            break

        # Find relevant context
        context = ""
        long_term_context = ""
        query_vec = genai.embed_content(model=EMBEDDING_MODEL, content=user_input)["embedding"]

        # Short-term context
        if embeddings:
            embeddings_arr = np.array(embeddings)
            sims = np.dot(embeddings_arr, query_vec) / (np.linalg.norm(embeddings_arr, axis=1) * np.linalg.norm(query_vec))
            top_idx = np.argsort(sims)[-3:][::-1]
            for i in top_idx:
                if 2*i+1 < len(chat.history):
                    context += f"[User]: {''.join(p.text for p in chat.history[2*i].parts)}\n[Model]: {''.join(p.text for p in chat.history[2*i+1].parts)}\n"

        # Long-term context
        if long_term_embeddings:
            lt_embeddings_arr = np.array(long_term_embeddings)
            lt_sims = np.dot(lt_embeddings_arr, query_vec) / (np.linalg.norm(lt_embeddings_arr, axis=1) * np.linalg.norm(query_vec))
            lt_top_idx = np.argsort(lt_sims)[-2:][::-1]  # top 2 long-term facts
            fact_keys = list(long_term["facts"].keys())
            for idx in lt_top_idx:
                if idx < len(fact_keys):
                    fact_key = fact_keys[idx]
                    fact_value = long_term["facts"][fact_key]
                    long_term_context += f"{fact_key}: {fact_value}\n"

        combined_context = f"Long-term memory:\n{long_term_context}\nConversation context:\n{context}" if long_term_context or context else ""
        prompt = f"{combined_context}\nUser: {user_input}" if combined_context else user_input

        response = chat.send_message(prompt, stream=True)
        print("Travis: ", end="")
        for chunk in response:
            try:
                text = chunk.text
                if text:
                    print(text, end="", flush=True)
            except ValueError:
                print(" [Response blocked due to safety reasons] ", end="", flush=True)
        print()

        # Extract and store long-term facts
        user_input_lower = user_input.lower()
        facts_updated = False

        # Name extraction
        match = re.search(r'my name is (.+?)(?:\n|$)', user_input, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            long_term["facts"]["user_name"] = name
            long_term_embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"user_name: {name}")["embedding"])
            facts_updated = True

        # Age extraction
        match = re.search(r'i am (\d+) years? old', user_input_lower)
        if match:
            age = match.group(1)
            long_term["facts"]["user_age"] = age
            long_term_embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"user_age: {age}")["embedding"])
            facts_updated = True

        # Location extraction
        match = re.search(r'i live in (.+?)(?:\n|$)', user_input_lower)
        if match:
            location = match.group(1).strip()
            long_term["facts"]["user_location"] = location
            long_term_embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"user_location: {location}")["embedding"])
            facts_updated = True

        # Occupation extraction
        match = re.search(r'i am a (.+?)(?:\n|$)', user_input_lower)
        if match:
            occupation = match.group(1).strip()
            long_term["facts"]["user_occupation"] = occupation
            long_term_embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"user_occupation: {occupation}")["embedding"])
            facts_updated = True

        # Hobbies/Interests extraction
        match = re.search(r'i like (.+?)(?:\n|$)', user_input_lower)
        if match:
            hobby = match.group(1).strip()
            long_term["facts"]["user_hobby"] = hobby
            long_term_embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"user_hobby: {hobby}")["embedding"])
            facts_updated = True

        # Birthdate extraction
        match = re.search(r'my birthdate is (.+?)(?:\n|$)', user_input_lower)
        if match:
            birthdate = match.group(1).strip()
            long_term["facts"]["user_birthdate"] = birthdate
            long_term_embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"user_birthdate: {birthdate}")["embedding"])
            facts_updated = True

        # Manual fact storage via chatbot command
        match = re.search(r'\b(remember|store)\b (.+)', user_input_lower)
        if match:
            fact = match.group(2).strip()
            fact_key = f"manual_{len(long_term['facts'])}"
            long_term["facts"][fact_key] = fact
            long_term_embeddings.append(
                genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=f"{fact_key}: {fact}"
                )["embedding"]
            )
            facts_updated = True

        if facts_updated:
            save_long_term_memory(long_term)
            np.savez_compressed(LONG_TERM_EMBEDDINGS_FILE, np.array(long_term_embeddings, dtype=object))

        # Update memory
        embeddings.append(genai.embed_content(model=EMBEDDING_MODEL, content=f"User: {user_input}\nTravis: {''.join(p.text for p in chat.history[-1].parts)}")["embedding"])
        save_memory(chat.history, embeddings)

if __name__ == "__main__":
    main()
