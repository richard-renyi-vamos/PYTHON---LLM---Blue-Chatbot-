import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blue Chatbot")
        self.root.geometry("400x500")
        self.root.config(bg="#3498db")

        self.chat_history = Text(root, bg="#ecf0f1", padx=10, pady=10, wrap="word")
        self.chat_history.pack(expand=True, fill="both")

        self.scrollbar = Scrollbar(root, command=self.chat_history.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_history.config(yscrollcommand=self.scrollbar.set)

        self.user_input = Entry(root, bg="#bdc3c7", font=("Helvetica", 12))
        self.user_input.pack(expand=True, fill="x", pady=10)

        self.send_button = Button(root, text="Send", command=self.send_message, bg="#2ecc71", fg="#ffffff")
        self.send_button.pack()

        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def send_message(self):
        user_message = self.user_input.get()
        self.display_message(f"You: {user_message}", "user")

        # Generate a response using the GPT-2 model
        input_ids = self.tokenizer.encode(user_message, return_tensors="pt")
        bot_response_ids = self.model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)
        bot_response = self.tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)

        self.display_message(f"Chatbot: {bot_response}", "bot")
        self.user_input.delete(0, "end")

    def display_message(self, message, sender):
        self.chat_history.insert("end", f"{message}\n")
        self.chat_history.yview("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
