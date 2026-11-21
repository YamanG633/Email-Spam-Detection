import re
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import joblib


# ==========================================================
# EMAIL SPAM DETECTOR - REFINED, PROPERLY ALIGNED GUI
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
except Exception as error:
    raise SystemExit(
        "Could not load the trained model files.\n"
        "Check that both files are inside the project's models folder.\n\n"
        f"Details: {error}"
    )


# ---------------------- Theme ----------------------

WINDOW_W = 920
WINDOW_H = 800

BG_TOP = "#6678E8"
BG_BOTTOM = "#7650B5"
PURPLE = "#6555D9"
PURPLE_HOVER = "#5544C7"
DARK = "#252B3A"
MUTED = "#778095"
WHITE = "#FFFFFF"
LIGHT = "#F5F6FA"
BORDER = "#DDE1F0"
GREEN = "#159B65"
GREEN_BG = "#E8F7EF"
RED = "#E85D67"
RED_BG = "#FFF0F0"


def hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def draw_gradient(canvas, width, height, top, bottom):
    top_rgb = hex_rgb(top)
    bottom_rgb = hex_rgb(bottom)

    for y in range(height):
        ratio = y / max(height - 1, 1)
        rgb = tuple(
            round(top_rgb[i] + (bottom_rgb[i] - top_rgb[i]) * ratio)
            for i in range(3)
        )
        canvas.create_line(0, y, width, y, fill=rgb_hex(rgb))


# ---------------------- ML functions ----------------------

def clean_email(email):
    email = str(email).lower()
    email = re.sub(r"[^a-zA-Z0-9\s]", " ", email)
    email = re.sub(r"\s+", " ", email)
    return email.strip()


def reset_result():
    result_label.config(
        text="Your classification result will appear here",
        fg="#687083",
        bg="#F1F2F7"
    )
    ham_value.config(text="--")
    spam_value.config(text="--")
    ham_bar.coords(ham_fill, 0, 0, 0, 14)
    spam_bar.coords(spam_fill, 0, 0, 0, 14)


def check_email():
    email = email_input.get("1.0", "end-1c").strip()

    if not email:
        messagebox.showwarning(
            "Email required",
            "Please enter or paste an email message first."
        )
        return

    try:
        features = tfidf.transform([clean_email(email)])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        probability_by_class = dict(zip(model.classes_, probabilities))

        ham_probability = probability_by_class.get(0, 0.0) * 100
        spam_probability = probability_by_class.get(1, 0.0) * 100

        if prediction == 1:
            result_label.config(
                text="⚠  SPAM EMAIL DETECTED",
                fg="#B42318",
                bg=RED_BG
            )
        else:
            result_label.config(
                text="✓  HAM  •  NORMAL EMAIL",
                fg="#067647",
                bg=GREEN_BG
            )

        ham_value.config(text=f"{ham_probability:.1f}%")
        spam_value.config(text=f"{spam_probability:.1f}%")

        bar_width = int(620 * ham_probability / 100)
        ham_bar.coords(ham_fill, 0, 0, bar_width, 14)

        bar_width = int(620 * spam_probability / 100)
        spam_bar.coords(spam_fill, 0, 0, bar_width, 14)

    except Exception as error:
        messagebox.showerror(
            "Classification error",
            f"The email could not be classified.\n\n{error}"
        )


sample_emails = [
    "Hello, please find the meeting notes attached. Let me know if you have any questions.",
    "CONGRATULATIONS! You have won a FREE cash prize. Claim your reward now by clicking this link!",
    "Your university examination timetable has been published. Please check the student portal."
]
sample_index = 0


def load_example():
    global sample_index
    email_input.delete("1.0", tk.END)
    email_input.insert("1.0", sample_emails[sample_index])
    sample_index = (sample_index + 1) % len(sample_emails)
    reset_result()
    email_input.focus_set()


def clear_email():
    email_input.delete("1.0", tk.END)
    reset_result()
    email_input.focus_set()


# ---------------------- Main window ----------------------

root = tk.Tk()
root.title("Email Spam Detection")
root.geometry(f"{WINDOW_W}x{WINDOW_H}")
root.resizable(False, False)
root.configure(bg=BG_TOP)

# Gradient background canvas
background = tk.Canvas(
    root,
    width=WINDOW_W,
    height=WINDOW_H,
    highlightthickness=0,
    bd=0
)
background.pack(fill="both", expand=True)
draw_gradient(background, WINDOW_W, WINDOW_H, BG_TOP, BG_BOTTOM)

# Card shadow and white card
background.create_rectangle(
    84, 55, 842, 755,
    fill="#51469A",
    outline=""
)
card = tk.Frame(root, bg=WHITE, bd=0, highlightthickness=0)
background.create_window(
    460, 400,
    window=card,
    width=748,
    height=690
)

# All card contents use normal Tkinter layout managers.
# This keeps elements aligned and prevents buttons overflowing.
content = tk.Frame(card, bg=WHITE)
content.pack(fill="both", expand=True, padx=42, pady=27)

# Header
header = tk.Frame(content, bg=WHITE)
header.pack(fill="x", pady=(0, 5))

title_row = tk.Frame(header, bg=WHITE)
title_row.pack(anchor="center")

tk.Label(
    title_row,
    text="✉",
    font=("Segoe UI Symbol", 25),
    bg=WHITE,
    fg=PURPLE
).pack(side="left", padx=(0, 10))

tk.Label(
    title_row,
    text="Email Spam Detector",
    font=("Segoe UI", 23, "bold"),
    bg=WHITE,
    fg=DARK
).pack(side="left")

tk.Label(
    header,
    text="Intelligent Email Classification System",
    font=("Segoe UI", 10),
    bg=WHITE,
    fg=MUTED
).pack(pady=(5, 17))

# Input label
tk.Label(
    content,
    text="Enter Your Email",
    font=("Segoe UI", 10, "bold"),
    bg=WHITE,
    fg="#343A49",
    anchor="w"
).pack(fill="x", pady=(0, 8))

# Text input container with subtle border
input_border = tk.Frame(
    content,
    bg=BORDER,
    padx=2,
    pady=2
)
input_border.pack(fill="x")

email_input = tk.Text(
    input_border,
    height=7,
    font=("Segoe UI", 11),
    wrap=tk.WORD,
    relief=tk.FLAT,
    bd=0,
    padx=12,
    pady=10,
    bg="#FFFFFF",
    fg="#303747",
    insertbackground=PURPLE,
    undo=True
)
email_input.pack(fill="both")

# Button row: three equal-width buttons inside the card.
button_row = tk.Frame(content, bg=WHITE)
button_row.pack(fill="x", pady=16)

for column in range(3):
    button_row.grid_columnconfigure(column, weight=1, uniform="buttons")

button_style = {
    "font": ("Segoe UI", 9, "bold"),
    "relief": tk.FLAT,
    "bd": 0,
    "cursor": "hand2",
    "height": 2,
    "width": 20,
    "activeforeground": WHITE
}

classify_button = tk.Button(
    button_row,
    text="CLASSIFY EMAIL",
    command=check_email,
    bg=PURPLE,
    fg=WHITE,
    activebackground=PURPLE_HOVER,
    **button_style
)
classify_button.grid(row=0, column=0, sticky="ew", padx=(0, 7), ipady=5)

examples_button = tk.Button(
    button_row,
    text="▣  LOAD EXAMPLES",
    command=load_example,
    bg="#ECEEF4",
    fg="#414756",
    activebackground="#DDE1EB",
    activeforeground="#414756",
    **{k: v for k, v in button_style.items() if k != "activeforeground"}
)
examples_button.grid(row=0, column=1, sticky="ew", padx=7, ipady=5)

clear_button = tk.Button(
    button_row,
    text="CLEAR",
    command=clear_email,
    bg="#FF6B70",
    fg=WHITE,
    activebackground="#E95761",
    **button_style
)
clear_button.grid(row=0, column=2, sticky="ew", padx=(7, 0), ipady=5)

# Result panel
result_panel = tk.Frame(content, bg="#F1F2F7", padx=12, pady=15)
result_panel.pack(fill="x", pady=(0, 20))

result_label = tk.Label(
    result_panel,
    text="Your classification result will appear here",
    font=("Segoe UI", 13, "bold"),
    bg="#F1F2F7",
    fg="#687083"
)
result_label.pack()

# Probability section
tk.Label(
    content,
    text="PREDICTION BREAKDOWN",
    font=("Segoe UI", 9, "bold"),
    bg=WHITE,
    fg="#737B8C",
    anchor="w"
).pack(fill="x", pady=(0, 12))

# Ham label and percentage
ham_row = tk.Frame(content, bg=WHITE)
ham_row.pack(fill="x")

tk.Label(
    ham_row,
    text="Ham (Normal)",
    font=("Segoe UI", 10, "bold"),
    bg=WHITE,
    fg=GREEN
).pack(side="left")

ham_value = tk.Label(
    ham_row,
    text="--",
    font=("Segoe UI", 10, "bold"),
    bg=WHITE,
    fg=GREEN
)
ham_value.pack(side="right")

ham_bar = tk.Canvas(
    content,
    height=14,
    bg="#E9ECF2",
    highlightthickness=0,
    bd=0
)
ham_bar.pack(fill="x", pady=(6, 14))
ham_fill = ham_bar.create_rectangle(
    0, 0, 0, 14,
    fill="#22B573",
    outline=""
)

# Spam label and percentage
spam_row = tk.Frame(content, bg=WHITE)
spam_row.pack(fill="x")

tk.Label(
    spam_row,
    text="Spam",
    font=("Segoe UI", 10, "bold"),
    bg=WHITE,
    fg=RED
).pack(side="left")

spam_value = tk.Label(
    spam_row,
    text="--",
    font=("Segoe UI", 10, "bold"),
    bg=WHITE,
    fg=RED
)
spam_value.pack(side="right")

spam_bar = tk.Canvas(
    content,
    height=14,
    bg="#E9ECF2",
    highlightthickness=0,
    bd=0
)
spam_bar.pack(fill="x", pady=(6, 0))
spam_fill = spam_bar.create_rectangle(
    0, 0, 0, 14,
    fill="#F16B73",
    outline=""
)

# Footer
tk.Label(
    content,
    text="Powered by TF-IDF  •  Multinomial Naive Bayes",
    font=("Segoe UI", 9),
    bg=WHITE,
    fg="#9AA1AF"
).pack(side="bottom", pady=(17, 0))

root.mainloop()
