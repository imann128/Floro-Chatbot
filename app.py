import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import random
import torch
import json
import re
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# --- COLORS ---
BG_COLOR = "#4E342E"       # Mocha brown
CHAT_BG = "#FFF8E1"        # Milk cream
BUTTON_COLOR = "#A1887F"   # Caramel
TEXT_COLOR = "#3E2723"     # Espresso
FONT = ("Comic Sans MS", 12)

# --- MAIN APP ---
class FloroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Floro ☕ - Your Café Assistant")
        self.root.config(bg=BG_COLOR)
        self.language = None
        self.language_selected = False
        self.user_name = None
        self.last_tag = None
        self.order_flow = {"items": {}, "confirmed": False, "total": 0}
        self.delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
        self.item_prices = {"tea": 110, "coffee": 180}
        self.bot_name = "Floro"

        # Load model and intents with error handling
        try:
            FILE = "data.pth"
            data = torch.load(FILE)
            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data["all_words"]
            self.tags = data["tags"]
            self.model_state = data["model_state"]

            with open("intents.json", "r", encoding="utf-8") as f:
                self.intents = json.load(f)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
            self.model.load_state_dict(self.model_state)
            self.model.eval()
        except FileNotFoundError:
            messagebox.showerror("Error", "Required files (data.pth or intents.json) not found.")
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
            self.root.destroy()
            return

        self.build_gui()
        self.display_message(self.bot_name, "Welcome to the café! ☕ Please choose a language: English or Urdu (type 'English' or 'Urdu')")

    def build_gui(self):
        # Header
        header = tk.Label(self.root, text="Floro ☕ - Your Café Assistant",
                         font=("Arial", 16, "bold"), bg=BG_COLOR, fg=CHAT_BG)
        header.pack(pady=10)

        # Chat area
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=15,
                                                  font=FONT, bg=CHAT_BG, fg=TEXT_COLOR,
                                                  bd=0, relief="flat")
        self.chat_area.pack(padx=10, pady=5)
        self.chat_area.config(state=tk.DISABLED)

        # Entry + Send
        entry_frame = tk.Frame(self.root, bg=BG_COLOR)
        entry_frame.pack(pady=10)

        self.user_input = tk.Entry(entry_frame, width=40, font=FONT, bg=CHAT_BG, fg=TEXT_COLOR)
        self.user_input.grid(row=0, column=0, padx=10)
        self.user_input.bind("<Return>", self.send_message)

        send_btn = tk.Button(entry_frame, text="Send", bg=BUTTON_COLOR, fg="white",
                            font=("Comic Sans MS", 12, "bold"), bd=0, padx=10, pady=5,
                            command=self.send_message)
        send_btn.grid(row=0, column=1)

    def send_message(self, event=None):
        msg = self.user_input.get().strip()
        if msg:
            self.display_message("You", msg)
            response = self.get_floro_response(msg)
            self.display_message(self.bot_name, response)
            self.user_input.delete(0, tk.END)

    def display_message(self, sender, msg):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: {msg}\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def show_order_summary(self):
        if not self.order_flow["items"]:
            return "You haven't added anything to your order yet." if self.language == "english" else "Aap ne abhi tak kuch order nahi kiya."

        summary = "Your order summary:\n" if self.language == "english" else "Apka order ka khulasa:\n"
        total = 0
        for item, quantity in self.order_flow["items"].items():
            item_total = self.item_prices[item] * quantity
            summary += f"- {item.capitalize()} x {quantity} = Rs. {item_total}\n"
            total += item_total
        summary += f"Total: Rs. {total}\n{'Do you want to confirm this order? (yes/no)' if self.language == 'english' else 'Kya aap yeh order confirm karna chahte hain? (yes/no)'}"
        self.order_flow["total"] = total
        return summary

    def get_floro_response(self, sentence):
        if not sentence:
            return "Please say something!" if self.language == "english" else "Kuch boliye!"

        if sentence.lower() == "close":
            self.root.destroy()
            return f"{'Goodbye' if self.language == 'english' else 'Khuda hafiz'}{f', {self.user_name}' if self.user_name else ''}!"

        # Handle initial language selection
        if not self.language_selected:
            if sentence.lower() == "urdu":
                self.language = "urdu"
                self.language_selected = True
                return "Urdu zuban chuni gayi hai."
            elif sentence.lower() == "english":
                self.language = "english"
                self.language_selected = True
                return "English language selected."
            else:
                return "Please choose 'English' or 'Urdu'."

        if "urdu" in sentence.lower() and ("speak" in sentence.lower() or "baat" in sentence.lower()):
            self.language = "urdu"
            return "Urdu zuban chuni gayi hai."

        if "english" in sentence.lower() and "language" in sentence.lower():
            self.language = "english"
            return "English language selected."

        if "summary" in sentence.lower() or "what have i ordered" in sentence.lower() or "kitna order kiya" in sentence.lower():
            return self.show_order_summary()

        if "how much" in sentence.lower() or "kitnay ki hai" in sentence.lower() or "kitnay ka hai" in sentence.lower():
            if self.order_flow["items"]:
                return self.show_order_summary()
            else:
                return f"{'Tea is Rs. 110 and coffee is Rs. 180.' if self.language == 'english' else 'Chai 110 rupay ki hai aur coffee 180 rupay ki hai.'}"

        if sentence.lower() == "yes" and self.order_flow["items"] and not self.order_flow["confirmed"]:
            self.order_flow["confirmed"] = True
            self.delivery_flow["asked"] = True
            return f"{'Great! Tell me your location' if self.language == 'english' else 'Zabardast! Apna address batayein'}"

        if sentence.lower() == "no" and self.order_flow["items"] and not self.order_flow["confirmed"]:
            self.order_flow = {"items": {}, "confirmed": False, "total": 0}
            return f"{'Order cancelled. Do you want anything else?' if self.language == 'english' else 'Order cancel ho gaya. Kuch aur chahiye?'}"

        if "do you deliver" in sentence.lower() or "delivery hai" in sentence.lower() or "delivery karte ho" in sentence.lower():
            self.last_tag = "delivery"
            self.delivery_flow["asked"] = True
            return f"{'Yes, we deliver to Islamabad, Lahore, and Karachi.' if self.language == 'english' else 'Jee haan, hum Islamabad, Lahore aur Karachi mein deliver karte hain.'}"

        if self.delivery_flow["asked"] and self.delivery_flow["location"] is None:
            if "i am in" in sentence.lower():
                location = sentence.lower().split("i am in")[-1].strip().capitalize()
            else:
                location = sentence.strip().capitalize()

            if location in ["Islamabad", "Lahore", "Karachi"]:
                self.delivery_flow["location"] = location
                return f"{'Yes, we deliver in ' if self.language == 'english' else 'Jee haan, hum '}{location}{'! Will it be an online payment or cash on delivery?' if self.language == 'english' else ' mein deliver karte hain! Online payment ya cash on delivery?'}"
            else:
                self.delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                return f"{'Sorry, we only deliver to Islamabad, Lahore, and Karachi.' if self.language == 'english' else 'Maaf kijiye, hum sirf Islamabad, Lahore aur Karachi mein deliver karte hain.'}"

        if self.delivery_flow["asked"] and self.delivery_flow["location"] and self.delivery_flow["payment"] is None:
            payment = None
            if "cash" in sentence.lower():
                payment = "cash on delivery"
            elif "online" in sentence.lower():
                payment = "online payment"

            if payment:
                self.delivery_flow["payment"] = payment
                if payment == "online payment":
                    return f"{'Will you be paying with Visa, Mastercard, or Paypak?' if self.language == 'english' else 'Kya aap Visa, Mastercard, ya Paypak se payment karenge?'}"
                else:  # Cash on delivery
                    if not self.order_flow["items"]:
                        self.delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                        return f"{'No items in your order. Please add items first.' if self.language == 'english' else 'Apkay order mein koi item nahi. Pehle items add karein.'}"
                    items_summary = ", ".join([f"{item.capitalize()} x{qty}" for item, qty in self.order_flow["items"].items()])
                    response = f"{'Order confirmed for ' if self.language == 'english' else 'Order confirm hua: '}{items_summary}.\n{'Total: Rs. ' if self.language == 'english' else 'Kul: Rs. '}{self.order_flow['total']} {'via' if self.language == 'english' else 'ke zariye'} {payment}.\n{'Your order will be delivered to ' if self.language == 'english' else 'Apka order '}{self.delivery_flow['location']}{'!' if self.language == 'english' else ' ko deliver hoga!'}{' Thank you!' if self.language == 'english' else ' Shukriya!'}"
                    self.delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                    self.order_flow = {"items": {}, "confirmed": False, "total": 0}
                    self.last_tag = None
                    return response
            else:
                return f"{'Please choose online payment or cash on delivery.' if self.language == 'english' else 'Online payment ya cash on delivery chunein.'}"

        if self.delivery_flow["asked"] and self.delivery_flow["location"] and self.delivery_flow["payment"] == "online payment" and self.delivery_flow["card_type"] is None:
            card_type = None
            if "visa" in sentence.lower():
                card_type = "Visa"
            elif "mastercard" in sentence.lower():
                card_type = "Mastercard"
            elif "paypak" in sentence.lower():
                card_type = "Paypak"

            if card_type:
                self.delivery_flow["card_type"] = card_type
                if not self.order_flow["items"]:
                    self.delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                    return f"{'No items in your order. Please add items first.' if self.language == 'english' else 'Apkay order mein koi item nahi. Pehle items add karein.'}"
                items_summary = ", ".join([f"{item.capitalize()} x{qty}" for item, qty in self.order_flow["items"].items()])
                response = f"{'Order confirmed for ' if self.language == 'english' else 'Order confirm hua: '}{items_summary}.\n{'Total: Rs. ' if self.language == 'english' else 'Kul: Rs. '}{self.order_flow['total']} {'via online payment with ' if self.language == 'english' else 'ke zariye online payment '} {card_type}.\n{'Your order will be delivered to ' if self.language == 'english' else 'Apka order '}{self.delivery_flow['location']}{'!' if self.language == 'english' else ' ko deliver hoga!'}{' Thank you!' if self.language == 'english' else ' Shukriya!'}"
                self.delivery_flow = {"asked": False, "location": None, "payment": None, "card_type": None}
                self.order_flow = {"items": {}, "confirmed": False, "total": 0}
                self.last_tag = None
                return response
            else:
                return f"{'Please choose Visa, Mastercard, or Paypak.' if self.language == 'english' else 'Visa, Mastercard, ya Paypak chunein.'}"

        if "my name is" in sentence.lower() or "mera naam" in sentence.lower():
            lowered = sentence.lower()
            name_start = lowered.find("my name is") + len("my name is") if "my name is" in lowered else lowered.find("mera naam") + len("mera naam")
            name_part = sentence[name_start:].strip()
            if name_part:
                self.user_name = name_part.capitalize()
                return f"{'Nice to meet you, ' if self.language == 'english' else 'Aapse mil kar khushi hui, '}{self.user_name}!"
            else:
                return f"{'Sorry, I didn\'t catch your name.' if self.language == 'english' else 'Maaf kijiye, mujhe apka naam nahi mila.'}"

        if "i am in" in sentence.lower() and not self.delivery_flow["asked"] and not self.order_flow["items"]:
            location = sentence.lower().split("i am in")[-1].strip().capitalize()
            if location in ["Islamabad", "Lahore", "Karachi"]:
                self.delivery_flow["asked"] = True
                self.delivery_flow["location"] = location
                self.last_tag = "delivery_confirmed"
                return f"{'Yes, we deliver in ' if self.language == 'english' else 'Jee haan, hum '}{location}{'! Will it be an online payment or cash on delivery?' if self.language == 'english' else ' mein deliver karte hain! Online payment ya cash on delivery?'}"
            else:
                return f"{'Sorry, we only deliver to Islamabad, Lahore, and Karachi.' if self.language == 'english' else 'Maaf kijiye, hum sirf Islamabad, Lahore aur Karachi mein deliver karte hain.'}"

        # NLU processing
        tokens = tokenize(sentence)
        X = bag_of_words(tokens, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.5:
            self.last_tag = tag
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    response_obj = random.choice(intent["responses"])
                    if isinstance(response_obj, dict):
                        response = response_obj.get(self.language, "Sorry, I don't have a response for this.")
                    else:
                        response = response_obj

                    if tag == "order_start" and not ("yes" in sentence.lower() or "yeah" in sentence.lower() or "sure" in sentence.lower()):
                        matches = re.findall(r'(\d+)?\s*(cups? of)?\s*(tea|coffee|chai)', sentence.lower())
                        if not matches:
                            matches = [(None, None, word) for word in ["tea", "coffee", "chai"] if word in sentence.lower()]
                        for quantity, _, item in matches:
                            item = "tea" if item == "chai" else item.lower()
                            quantity = int(quantity) if quantity else 1
                            if item in self.item_prices:
                                self.order_flow["items"][item] = self.order_flow["items"].get(item, 0) + quantity
                                self.order_flow["total"] += self.item_prices[item] * quantity
                        if self.order_flow["items"]:
                            items_summary = ", ".join([f"{itm.capitalize()} x{q}" for itm, q in self.order_flow["items"].items()])
                            response = f"{'Added to your order: ' if self.language == 'english' else 'Apkay order mein shamil: '}{items_summary}. {'Would you like to confirm this order? (yes/no)' if self.language == 'english' else 'Kya aap yeh order confirm karna chahte hain? (yes/no)'}"
                        else:
                            response = f"{'Sure, what would you like to order?' if self.language == 'english' else 'Zabardast, aap kya order karna chahte hain?'}"

                    elif tag == "items":
                        matches = re.findall(r'(\d+)?\s*(cups? of)?\s*(tea|coffee|chai)', sentence.lower())
                        if not matches:
                            matches = [(None, None, word) for word in ["tea", "coffee", "chai"] if word in sentence.lower()]
                        for quantity, _, item in matches:
                            item = "tea" if item == "chai" else item.lower()
                            quantity = int(quantity) if quantity else 1
                            if item in self.item_prices:
                                self.order_flow["items"][item] = self.order_flow["items"].get(item, 0) + quantity
                                self.order_flow["total"] += self.item_prices[item] * quantity
                        if self.order_flow["items"]:
                            items_summary = ", ".join([f"{itm.capitalize()} x{q}" for itm, q in self.order_flow["items"].items()])
                            response = f"{'Added to your order: ' if self.language == 'english' else 'Apkay order mein shamil: '}{items_summary}. {'Would you like to confirm this order? (yes/no)' if self.language == 'english' else 'Kya aap yeh order confirm karna chahte hain? (yes/no)'}"

                    if tag == "delivery":
                        self.delivery_flow["asked"] = True

                    if isinstance(response, str) and "{name}" in response:
                        name_to_use = self.user_name if self.user_name else "friend"
                        response = response.replace("{name}", name_to_use)

                    return response

        return f"{'I\'m sorry, I didn\'t understand that.' if self.language == 'english' else 'Maaf kijiye, mujhe samajh nahi aaya.'}"

# --- RUN APP ---
if __name__ == '__main__':
    root = tk.Tk()
    app = FloroApp(root)
    root.mainloop()
