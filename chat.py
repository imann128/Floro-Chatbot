import random
import torch
import json
import re
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

language = None  # Start with no language selected

FILE = "data.pth"
try:
    data = torch.load(FILE)
except FileNotFoundError:
    print("Error: data.pth file not found.")
    exit()

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('intents.json', 'r', encoding="utf-8") as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: intents.json file not found.")
    exit()

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Floro"
user_name = None
last_tag = None
language_selected = False

# Initial language prompt
print(f"{bot_name}: Welcome to the café! ☕ Please choose a language: English or Urdu (type 'English' or 'Urdu')")

delivery_flow = {
    "asked": False,
    "location": None,
    "payment": None,
    "card_type": None  # New state for card type
}

item_prices = {
    "tea": 110,
    "coffee": 180
}

order_flow = {
    "items": {},
    "confirmed": False,
    "total": 0
}

def show_order_summary():
    if not order_flow["items"]:
        return "You haven't added anything to your order yet." if language == "english" else "Aap ne abhi tak kuch order nahi kiya."

    summary = "Your order summary:\n" if language == "english" else "Apka order ka khulasa:\n"
    total = 0
    for item, quantity in order_flow["items"].items():
        item_total = item_prices[item] * quantity
        summary += f"- {item.capitalize()} x {quantity} = Rs. {item_total}\n"
        total += item_total
    summary += f"Total: Rs. {total}\n{'Do you want to confirm this order? (yes/no)' if language == 'english' else 'Kya aap yeh order confirm karna chahte hain? (yes/no)'}"
    order_flow["total"] = total
    return summary

while True:
    sentence = input("You: ").strip()

    if not sentence:
        print(f"{bot_name}: {'Please say something!' if language == 'english' else 'Kuch boliye!'}")
        continue

    if sentence.lower() == "close":
        print(f"{bot_name}: {'Goodbye' if language == 'english' else 'Khuda hafiz'}{f', {user_name}' if user_name else ''}!")
        break

    # Handle initial language selection
    if not language_selected:
        if sentence.lower() == "urdu":
            language = "urdu"
            language_selected = True
            print(f"{bot_name}: Urdu zuban chuni gayi hai.")
            continue
        elif sentence.lower() == "english":
            language = "english"
            language_selected = True
            print(f"{bot_name}: English language selected.")
            continue
        else:
            print(f"{bot_name}: Please choose 'English' or 'Urdu'.")
            continue

    if "urdu" in sentence.lower() and ("speak" in sentence.lower() or "baat" in sentence.lower()):
        language = "urdu"
        print(f"{bot_name}: Urdu zuban chuni gayi hai.")
        continue

    if "english" in sentence.lower() and "language" in sentence.lower():
        language = "english"
        print(f"{bot_name}: English language selected.")
        continue

    if "summary" in sentence.lower() or "what have i ordered" in sentence.lower() or "kitna order kiya" in sentence.lower():
        print(f"{bot_name}: {show_order_summary()}")
        continue

    # Handles price queries
    if "how much" in sentence.lower() or "kitnay ki hai" in sentence.lower() or "kitnay ka hai" in sentence.lower():
        if order_flow["items"]:
            print(f"{bot_name}: {show_order_summary()}")
        else:
            print(f"{bot_name}: {'Tea is Rs. 110 and coffee is Rs. 180.' if language == 'english' else 'Chai 110 rupay ki hai aur coffee 180 rupay ki hai.'}")
        continue

    # Checks for location only if user confirms order
    if sentence.lower() == "yes" and order_flow["items"] and not order_flow["confirmed"]:
        order_flow["confirmed"] = True
        print(f"{bot_name}: {'Great! Tell me your location' if language == 'english' else 'Zabardast! Apna address batayein'}")
        delivery_flow["asked"] = True
        continue

    # If user cancels order
    if sentence.lower() == "no" and order_flow["items"] and not order_flow["confirmed"]:
        print(f"{bot_name}: {'Order cancelled. Do you want anything else?' if language == 'english' else 'Order cancel ho gaya. Kuch aur chahiye?'}")
        order_flow = {"items": {}, "confirmed": False, "total": 0}
        continue

    # Handles delivery queries for example: "Do you deliver?"
    if "do you deliver" in sentence.lower() or "delivery hai" in sentence.lower() or "delivery karte ho" in sentence.lower():
        print(f"{bot_name}: {'Yes, we deliver to Islamabad, Lahore, and Karachi.' if language == 'english' else 'Jee haan, hum Islamabad, Lahore aur Karachi mein deliver karte hain.'}")
        last_tag = "delivery"
        delivery_flow["asked"] = True
        continue

    # Asks the user for their location
    if delivery_flow["asked"] and delivery_flow["location"] is None:
        if "i am in" in sentence.lower():
            location = sentence.lower().split("i am in")[-1].strip().capitalize()
        else:
            location = sentence.strip().capitalize()

        # Checks if user lives in a place where user lives
        if location in ["Islamabad", "Lahore", "Karachi"]:
            delivery_flow["location"] = location
            print(f"{bot_name}: {'Yes, we deliver in ' if language == 'english' else 'Jee haan, hum '}{location}{'! Will it be an online payment or cash on delivery?' if language == 'english' else ' mein deliver karte hain! Online payment ya cash on delivery?'}")
        else:
            print(f"{bot_name}: {'Sorry, we only deliver to Islamabad, Lahore, and Karachi.' if language == 'english' else 'Maaf kijiye, hum sirf Islamabad, Lahore aur Karachi mein deliver karte hain.'}")
            delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
        continue

    # Handles payment workflow
    if delivery_flow["asked"] and delivery_flow["location"] and delivery_flow["payment"] is None:
        payment = None
        if "cash" in sentence.lower():
            payment = "cash on delivery"
        elif "online" in sentence.lower():
            payment = "online payment"

        # Checks if user has an accepted payment gateway
        if payment:
            delivery_flow["payment"] = payment
            if payment == "online payment":
                print(f"{bot_name}: {'Will you be paying with Visa, Mastercard, or Paypak?' if language == 'english' else 'Kya aap Visa, Mastercard, ya Paypak se payment karenge?'}")
                continue
            else:  # Cash on delivery
                if not order_flow["items"]:
                    print(f"{bot_name}: {'No items in your order. Please add items first.' if language == 'english' else 'Apkay order mein koi item nahi. Pehle items add karein.'}")
                    delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                    continue
                items_summary = ", ".join([f"{item.capitalize()} x{qty}" for item, qty in order_flow["items"].items()])
                print(f"{bot_name}: {'Order confirmed for ' if language == 'english' else 'Order confirm hua: '}{items_summary}.\n{'Total: Rs. ' if language == 'english' else 'Kul: Rs. '}{order_flow['total']} {'via' if language == 'english' else 'ke zariye'} {payment}.\n{'Your order will be delivered to ' if language == 'english' else 'Apka order '}{delivery_flow['location']}{'!' if language == 'english' else ' ko deliver hoga!'}{' Thank you!' if language == 'english' else ' Shukriya!'}")
                delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                order_flow = {"items": {}, "confirmed": False, "total": 0}
                last_tag = None
                continue
        else:
            print(f"{bot_name}: {'Please choose online payment or cash on delivery.' if language == 'english' else 'Online payment ya cash on delivery chunein.'}")
            continue
    
    # Credit card validation
    if delivery_flow["asked"] and delivery_flow["location"] and delivery_flow["payment"] == "online payment" and delivery_flow["card_type"] is None:
        card_type = None
        if "visa" in sentence.lower():
            card_type = "Visa"
        elif "mastercard" in sentence.lower():
            card_type = "Mastercard"
        elif "paypak" in sentence.lower():
            card_type = "Paypak"

        # If user tries to place an order without any items
        if card_type:
            delivery_flow["card_type"] = card_type
            if not order_flow["items"]:
                print(f"{bot_name}: {'No items in your order. Please add items first.' if language == 'english' else 'Apkay order mein koi item nahi. Pehle items add karein.'}")
                delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                continue
            items_summary = ", ".join([f"{item.capitalize()} x{qty}" for item, qty in order_flow["items"].items()])
            print(f"{bot_name}: {'Order confirmed for ' if language == 'english' else 'Order confirm hua: '}{items_summary}.\n{'Total: Rs. ' if language == 'english' else 'Kul: Rs. '}{order_flow['total']} {'via online payment with ' if language == 'english' else 'ke zariye online payment '} {card_type}.\n{'Your order will be delivered to ' if language == 'english' else 'Apka order '}{delivery_flow['location']}{'!' if language == 'english' else ' ko deliver hoga!'}{' Thank you!' if language == 'english' else ' Shukriya!'}")
            delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
            order_flow = {"items": {}, "confirmed": False, "total": 0}
            last_tag = None
            continue
        else:
            print(f"{bot_name}: {'Please choose Visa, Mastercard, or Paypak.' if language == 'english' else 'Visa, Mastercard, ya Paypak chunein.'}")
            continue

    # If user tries to deliver from a location where the restaurant doesn't deliver
    if "i am in" in sentence.lower() and not delivery_flow["asked"] and not order_flow["items"]:
        location = sentence.lower().split("i am in")[-1].strip().capitalize()
        if location in ["Islamabad", "Lahore", "Karachi"]:
            print(f"{bot_name}: {'Yes, we deliver in ' if language == 'english' else 'Jee haan, hum '}{location}{'. Would you like to place an order?' if language == 'english' else '. Kya aap order dena chahte hain?'}")
        else:
            print(f"{bot_name}: {'Sorry, we only deliver to Islamabad, Lahore, and Karachi.' if language == 'english' else 'Maaf kijiye, hum sirf Islamabad, Lahore aur Karachi mein deliver karte hain.'}")
        continue

    # NLU: tokenize and classify
    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Low threshold allows chatbot to respond with confidence
    if prob.item() > 0.5:
        last_tag = tag
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response_obj = random.choice(intent["responses"])
                if isinstance(response_obj, dict):
                    response = response_obj.get(language, "Sorry, I don't have a response for this.")
                else:
                    response = response_obj

                # Order starte after user confirmation
                if tag == "order_start" and not ("yes" in sentence.lower() or "yeah" in sentence.lower() or "sure" in sentence.lower()):
                    matches = re.findall(r'(\d+)?\s*(cups? of)?\s*(tea|coffee|chai)', sentence.lower()) # checks keywords 
                    if not matches:
                        matches = [(None, None, word) for word in ["tea", "coffee", "chai"] if word in sentence.lower()]
                    for quantity, _, item in matches:
                        item = "tea" if item == "chai" else item.lower()
                        quantity = int(quantity) if quantity else 1
                        if item in item_prices:
                            order_flow["items"][item] = order_flow["items"].get(item, 0) + quantity
                            order_flow["total"] += item_prices[item] * quantity   # Total order quantity
                    if order_flow["items"]:  # Chatbot checks with tags in the 'intents' file and answers accordingly
                        items_summary = ", ".join([f"{itm.capitalize()} x{q}" for itm, q in order_flow["items"].items()])
                        response = f"{'Added to your order: ' if language == 'english' else 'Apkay order mein shamil: '}{items_summary}. {'Would you like to confirm this order? (yes/no)' if language == 'english' else 'Kya aap yeh order confirm karna chahte hain? (yes/no)'}"
                    else:
                        response = f"{'Sure, what would you like to order?' if language == 'english' else 'Zabardast, aap kya order karna chahte hain?'}"

                elif tag == "items":
                    matches = re.findall(r'(\d+)?\s*(cups? of)?\s*(tea|coffee|chai)', sentence.lower())
                    if not matches:
                        matches = [(None, None, word) for word in ["tea", "coffee", "chai"] if word in sentence.lower()]
                    for quantity, _, item in matches:
                        item = "tea" if item == "chai" else item.lower()
                        quantity = int(quantity) if quantity else 1
                        if item in item_prices:
                            order_flow["items"][item] = order_flow["items"].get(item, 0) + quantity # Handles quantity changes
                            order_flow["total"] += item_prices[item] * quantity
                    if order_flow["items"]:
                        items_summary = ", ".join([f"{itm.capitalize()} x{q}" for itm, q in order_flow["items"].items()]) # prints order summary
                        response = f"{'Added to your order: ' if language == 'english' else 'Apkay order mein shamil: '}{items_summary}. {'Would you like to confirm this order? (yes/no)' if language == 'english' else 'Kya aap yeh order confirm karna chahte hain? (yes/no)'}"

                if tag == "delivery":
                    delivery_flow["asked"] = True

                # If user doesn't provide a name, refer to them as "friend"
                if isinstance(response, str) and "{name}" in response:
                    name_to_use = user_name if user_name else "friend"
                    response = response.replace("{name}", name_to_use)

                print(f"{bot_name}: {response}")
                break
    else: 
        print(f"{bot_name}: {'I\'m sorry, I didn\'t understand that.' if language == 'english' else 'Maaf kijiye, mujhe samajh nahi aaya.'}") # Fallback logic
