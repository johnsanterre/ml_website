"""
================================================================================
 FLAVORCART  —  ML2 FINAL PROJECT
 Your first job as a Machine Learning Engineer
================================================================================

 THE STORY
 ---------
 You have just been hired as the first (and only!) ML engineer at FlavorCart,
 a small food-delivery startup. The company has data: customers, dishes,
 orders, written reviews, dish photos, and a folder of help documents.
 What it does NOT have is anyone who can turn that data into something useful.

 That is your job now.

 Over 10 assignments your boss will hand you one real problem at a time.
 Each assignment builds on the ones before it — the recommender you build in
 Assignment 3 reuses the data work from Assignment 1, and the chatbot in
 Assignment 9 reuses the word embeddings from Assignment 5. By the end you
 will have built, with your own hands, a miniature version of almost every
 major system covered in ML2.

 THE ROADMAP                                          (▉ = harder)
 ----------------------------------------------------------------------------
  #   Title                            Course topic              Difficulty
  1   Meet the data                    Intro to deep learning    ▉░░░░░░░░░
  2   Predict delivery ratings         Training real models      ▉▉░░░░░░░░
  3   "Customers who like X..."        Vectors & similarity      ▉▉▉░░░░░░░
  4   Find the weird orders            Autoencoders              ▉▉▉▉░░░░░░
  5   What does "spicy" mean?          Word embeddings           ▉▉▉▉▉░░░░░
  6   Read reviews like a model        Sequences & attention     ▉▉▉▉▉▉░░░░
  7   Is that photo really a pizza?    CNNs                      ▉▉▉▉▉▉▉░░░
  8   Autocomplete the review          Generative models / LLMs  ▉▉▉▉▉▉▉▉░░
  9   The support chatbot              RAG                       ▉▉▉▉▉▉▉▉▉░
 10   Ship it (capstone)               Evaluation & agents       ▉▉▉▉▉▉▉▉▉▉
 ----------------------------------------------------------------------------
 Suggested pace: about one assignment per week. 1-2 are lighter; 9-10 heavier.

 HOW TO WORK  (read this twice!)
 -------------------------------
 1. You only ever run ONE command:

        python ml2_final_project.py 1        <- run assignment 1
        python ml2_final_project.py 2        <- run assignment 2  ... and so on
        python ml2_final_project.py progress <- see how far you've come

 2. Each assignment is a section of this file. Scroll to it. Read the STORY
    and the GOAL. Then work through the numbered STEPS from top to bottom.

 3. Your work happens ONLY on lines marked  TODO . They look like this:

        n_customers = None   # TODO[1.1]: replace None with ...

    Replace the placeholder, save the file, and run the assignment again.
    NEVER edit code that is not marked TODO (you can't break anything else).

 4. After every step the program runs a CHECKPOINT. A green check means you
    got it — keep going. A red cross comes with a hint. The program always
    stops at the first thing that isn't finished, so you always know exactly
    what to do next: fix the one thing it points at.

 5. Questions marked PAUSE AND THINK are answered by typing a sentence or two
    into the quoted string right below the question. They are checked for
    effort (length), not for a "right answer" — write what you actually think.

 RULES OF THE GAME
 -----------------
 - No math background needed. Every formula that appears is also explained
   in plain English right next to it.
 - You need Python 3.9+, and two libraries:  numpy  and  torch
       pip install numpy torch
 - Everything is generated inside this file with a fixed random seed:
   no downloads, no API keys, no internet. Your results are reproducible —
   if a checkpoint passed yesterday, it will pass today.
 - Stuck for more than 30 minutes on one step? Write down what you tried in
   the THINK string, skip nothing, and ask your instructor. That's what a
   real junior engineer would do.

================================================================================
"""

import json
import math
import os
import sys

# ------------------------------------------------------------------------
# Friendly dependency check (before anything else can crash confusingly)
# ------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    sys.exit("Missing library: numpy.  Fix:  pip install numpy")
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("Missing library: torch.  Fix:  pip install torch")


# ==========================================================================
# ENGINE ROOM — checkpoint machinery. You never edit anything in this block,
# but you're welcome to read it; it's ~100 lines of ordinary Python.
# ==========================================================================

SEED = 42
_QUIET = False           # True while an assignment is re-run as a dependency
ARTIFACTS = {}           # finished assignments park their outputs here
_PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".flavorcart_progress.json")


class NotDoneYet(Exception):
    """Raised to stop an assignment at the first unfinished/failed step."""


def say(*lines):
    """Print story/instruction text (silenced during dependency re-runs)."""
    if not _QUIET:
        for line in lines:
            print(line)


def banner(title):
    say("", "=" * 74, "  " + title, "=" * 74)


def need_value(value, tag, hint):
    """Stop (kindly) if a TODO placeholder is still None."""
    if value is None:
        say("",
            f"  ⏸  TODO[{tag}] is still waiting for you.",
            f"     {hint}",
            f"     Open this file, search for  TODO[{tag}]  , edit that line,",
            f"     save, and run this assignment again.")
        raise NotDoneYet(tag)
    return value


def checkpoint(tag, condition, ok_msg, hint):
    """Green check if condition holds; otherwise a hint and a full stop."""
    if condition:
        say(f"  ✅ CHECKPOINT {tag}: {ok_msg}")
    else:
        say(f"  ❌ CHECKPOINT {tag}: not quite.",
            f"     Hint: {hint}",
            "     Fix it and run the assignment again — it restarts in seconds.")
        raise NotDoneYet(tag)


def think_check(tag, answer):
    """PAUSE-AND-THINK answers just need honest effort (~ a sentence)."""
    if answer is None or len(answer.strip()) < 25:
        say("",
            f"  ✍️  PAUSE AND THINK [{tag}]: write your answer (a sentence or",
            f"     two, in your own words) into the THINK_{tag.replace('.', '_')} string, then rerun.")
        raise NotDoneYet(tag)
    say(f"  ✅ THINK {tag}: answer recorded ({len(answer.split())} words).")


def _load_progress():
    try:
        with open(_PROGRESS_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _mark_done(n):
    prog = _load_progress()
    prog[str(n)] = True
    with open(_PROGRESS_FILE, "w") as fh:
        json.dump(prog, fh)


def show_progress():
    prog = _load_progress()
    banner("YOUR PROGRESS AT FLAVORCART")
    titles = ["Meet the data", "Predict delivery ratings",
              "Customers who like X also like Y", "Find the weird orders",
              "What does 'spicy' mean to a computer?",
              "Read reviews like a model", "Is that photo really a pizza?",
              "Autocomplete the review", "The support chatbot",
              "Ship it (capstone)"]
    for i, t in enumerate(titles, 1):
        mark = "✅" if prog.get(str(i)) else "⬜"
        say(f"   {mark}  Assignment {i:>2}: {t}")
    done = sum(1 for i in range(1, 11) if prog.get(str(i)))
    say("", f"   {done}/10 complete." +
        ("  🎉 You shipped the whole thing!" if done == 10 else
         f"  Next up:  python ml2_final_project.py {done + 1}"))


def need(n, build_fn):
    """Re-run an earlier assignment quietly because a later one needs it."""
    global _QUIET
    if n in ARTIFACTS:
        return ARTIFACTS[n]
    was_quiet, _QUIET = _QUIET, True
    try:
        build_fn()
    except (NotDoneYet, NotImplementedError):
        _QUIET = was_quiet
        say("",
            f"  🔗 This assignment builds on Assignment {n}, which isn't",
            f"     finished yet. Run:  python ml2_final_project.py {n}")
        raise NotDoneYet(f"needs A{n}")
    _QUIET = was_quiet
    return ARTIFACTS[n]


# ==========================================================================
# THE FLAVORCART DATA WORLD
# --------------------------------------------------------------------------
# Everything below generates the company's data from a fixed random seed.
# You never edit this block either — but DO skim it once: knowing what the
# data looks like is half of every real ML job.
# ==========================================================================

CUISINES = ["italian", "mexican", "japanese", "indian", "thai", "american"]

INGREDIENTS = ["tomato", "cheese", "basil", "beef", "chicken", "pork", "tofu",
               "rice", "noodles", "beans", "corn", "avocado", "fish", "shrimp",
               "seaweed", "curry", "chili", "coconut", "peanut", "lime",
               "onion", "garlic", "bread", "lettuce"]

_DISH_NAMES = {
    "italian":  ["Margherita Pizza", "Beef Lasagna", "Spaghetti Carbonara",
                 "Mushroom Risotto", "Pepperoni Pizza", "Chicken Parmesan",
                 "Four Cheese Gnocchi", "Caprese Salad", "Tomato Bruschetta",
                 "Fettuccine Alfredo"],
    "mexican":  ["Beef Tacos", "Chicken Burrito", "Veggie Quesadilla",
                 "Pork Carnitas Bowl", "Loaded Nachos", "Shrimp Fajitas",
                 "Chicken Enchiladas", "Guacamole & Chips", "Bean Tostada",
                 "Carne Asada Plate"],
    "japanese": ["Salmon Sushi Roll", "Tuna Sashimi Set", "Chicken Teriyaki",
                 "Pork Ramen", "Veggie Tempura", "Spicy Tuna Roll",
                 "Beef Udon", "Miso Soup Set", "California Roll",
                 "Eel Rice Bowl"],
    "indian":   ["Butter Chicken", "Chana Masala", "Lamb Vindaloo",
                 "Palak Paneer", "Chicken Tikka Masala", "Vegetable Biryani",
                 "Garlic Naan Combo", "Dal Tadka", "Beef Madras",
                 "Tandoori Chicken"],
    "thai":     ["Pad Thai", "Green Curry Chicken", "Tom Yum Soup",
                 "Basil Fried Rice", "Massaman Curry", "Papaya Salad",
                 "Red Curry Shrimp", "Pineapple Fried Rice", "Drunken Noodles",
                 "Coconut Soup"],
    "american": ["Classic Cheeseburger", "BBQ Pulled Pork Sandwich",
                 "Buffalo Wings", "Caesar Salad", "Mac and Cheese",
                 "Philly Cheesesteak", "Cobb Salad", "Fried Chicken Basket",
                 "Turkey Club", "Loaded Baked Potato"],
}

_CUISINE_INGREDIENTS = {   # which ingredients each cuisine usually uses
    "italian":  ["tomato", "cheese", "basil", "beef", "chicken", "garlic", "bread"],
    "mexican":  ["beef", "chicken", "pork", "beans", "corn", "avocado", "chili", "lime", "cheese"],
    "japanese": ["fish", "shrimp", "rice", "seaweed", "noodles", "tofu", "pork"],
    "indian":   ["chicken", "curry", "chili", "rice", "tofu", "onion", "garlic", "beef"],
    "thai":     ["noodles", "curry", "chili", "coconut", "peanut", "lime", "shrimp", "chicken", "rice"],
    "american": ["beef", "cheese", "chicken", "bread", "lettuce", "corn", "pork"],
}

_FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Riley", "Morgan",
          "Jamie", "Quinn", "Avery", "Dana", "Reese", "Skyler", "Drew",
          "Emerson", "Finley", "Harper", "Kendall", "Logan", "Parker"]
_LAST = ["Lee", "Garcia", "Chen", "Patel", "Kim", "Nguyen", "Lopez",
         "Smith", "Johnson", "Okafor"]

FAQ_DOCS = [
    ("refunds", "Refund Policy",
     "If your order arrives cold damaged or wrong you can request a full "
     "refund within 24 hours from the order page. Refunds are returned to "
     "your original payment method within 3 to 5 business days."),
    ("delivery_time", "Delivery Times",
     "Typical delivery takes 30 to 45 minutes. During peak dinner hours "
     "delivery can take up to 60 minutes. You will receive a text message "
     "when your courier is nearby."),
    ("delivery_area", "Delivery Areas",
     "FlavorCart currently delivers within the downtown core and the "
     "university district. Enter your address at checkout to confirm your "
     "building is inside our delivery zone."),
    ("allergens", "Allergen Information",
     "Every dish page lists major allergens including peanut shellfish "
     "dairy gluten and soy. Use the allergen filter in search to hide "
     "dishes that contain an allergen you must avoid."),
    ("vegan", "Vegan and Vegetarian Options",
     "Use the vegan filter to see plant based dishes such as tofu curry "
     "veggie tempura and bean tostada. Vegetarian dishes may contain dairy "
     "or egg and are labeled separately from vegan dishes."),
    ("spice", "Spice Levels",
     "Dishes are rated from mild to three chilis. If you are sensitive to "
     "spicy food choose dishes rated mild or one chili. If you love spicy "
     "food look for the three chili label. Extra spicy chili can be added "
     "to many thai and indian dishes at checkout."),
    ("payment", "Payment Methods",
     "We accept credit cards debit cards and mobile wallets. Cash is not "
     "accepted. Your card is charged when the restaurant confirms your "
     "order not when you place it."),
    ("tipping", "Tipping Your Courier",
     "Tips go entirely to your courier. You can add a tip at checkout or "
     "increase it up to one hour after delivery from the order page."),
    ("tracking", "Tracking Your Order",
     "After checkout open the order page to track your order with live "
     "status updates: confirmed cooking picked up and delivered. A map "
     "lets you track your courier once the order is picked up. You can "
     "track every current and past order from the orders tab."),
    ("cancel", "Cancelling an Order",
     "You can cancel free of charge until the restaurant starts cooking. "
     "To cancel open the order page and tap cancel order. After cooking "
     "begins a cancel fee equal to half the order value applies because "
     "the food is already being prepared."),
    ("group", "Group Orders",
     "Start a group order and share the link with friends. Everyone adds "
     "their own dishes and you check out together with one delivery fee "
     "split between participants."),
    ("loyalty", "FlavorPoints Loyalty Program",
     "You earn one FlavorPoint per dollar spent. Every 200 points converts "
     "into a 10 dollar credit automatically applied to your next order."),
]

# Review-text templates, grouped by star rating. Slots: {dish} {ing}
_REVIEW_TEMPLATES = {
    5: ["absolutely loved the {dish} the {ing} tasted fresh and the flavor was amazing",
        "the {dish} was delicious hot and perfectly cooked will definitely order again",
        "amazing meal the {ing} was wonderful and delivery arrived fast and warm",
        "best {dish} in town rich flavor generous portion five stars"],
    4: ["really enjoyed the {dish} the {ing} was tasty and it arrived warm",
        "good {dish} nice flavor and friendly courier just slightly small portion",
        "the {dish} was very good fresh {ing} and quick delivery",
        "solid meal the {dish} tasted great though the packaging leaked a little"],
    3: ["the {dish} was okay the {ing} was fine but nothing special",
        "average {dish} decent portion but the flavor was a bit bland",
        "not bad not great the {dish} arrived warm but tasted ordinary",
        "the {dish} was acceptable though i expected more {ing} for the price"],
    2: ["disappointed the {dish} arrived cold and the {ing} was chewy",
        "the {dish} took forever and tasted stale barely warm on arrival",
        "mediocre {dish} soggy texture and the {ing} had almost no flavor",
        "underwhelming the {dish} was cold late and smaller than expected"],
    1: ["terrible the {dish} was cold soggy and inedible i want a refund",
        "awful experience the {ing} smelled off and delivery was an hour late",
        "worst {dish} ever freezing cold missing items and rude courier",
        "horrible the {dish} arrived ruined spilled everywhere total waste of money"],
}
_SPICY_TAIL = " the {level} spicy kick was {verdict}"


def build_world():
    """Generate FlavorCart's entire dataset. Same seed -> same data, always."""
    rng = np.random.default_rng(SEED)

    # ----- dishes -----------------------------------------------------------
    dishes = []
    for cuisine in CUISINES:
        for name in _DISH_NAMES[cuisine]:
            typical = _CUISINE_INGREDIENTS[cuisine]
            n_ing = int(rng.integers(3, 6))
            chosen = list(rng.choice(typical, size=min(n_ing, len(typical)),
                                     replace=False))
            vec = np.zeros(len(INGREDIENTS), dtype=np.float32)
            for ing in chosen:
                vec[INGREDIENTS.index(ing)] = 1.0
            spice = int(np.clip(rng.poisson(1.6 if cuisine in ("thai", "indian", "mexican") else 0.5), 0, 3))
            dishes.append({
                "dish_id": len(dishes),
                "name": name,
                "cuisine": cuisine,
                "price": float(np.round(rng.uniform(8, 24), 2)),
                "spice": spice,                        # 0 (mild) .. 3 (🔥🔥🔥)
                "veg": bool(vec[[INGREDIENTS.index(m) for m in
                                 ("beef", "chicken", "pork", "fish", "shrimp")]].sum() == 0),
                "ingredients": vec,                    # 24 numbers of 0.0/1.0
            })

    # ----- customers ---------------------------------------------------------
    customers = []
    for cid in range(200):
        customers.append({
            "customer_id": cid,
            "name": f"{_FIRST[cid % len(_FIRST)]} {_LAST[(cid // len(_FIRST)) % len(_LAST)]}",
            "favorite_cuisine": CUISINES[int(rng.integers(0, len(CUISINES)))],
            "spice_tolerance": int(rng.integers(0, 4)),   # 0..3
            "budget": float(np.round(rng.uniform(10, 26), 2)),
            "weeks_active": int(rng.integers(1, 105)),
        })

    def affinity(cust, dish):
        """Hidden 'true' taste score used to simulate ratings (1..5)."""
        score = 3.65
        score += 1.0 if dish["cuisine"] == cust["favorite_cuisine"] else -0.2
        score -= 0.38 * abs(dish["spice"] - cust["spice_tolerance"])
        score -= 0.08 * max(0.0, dish["price"] - cust["budget"])
        return float(np.clip(score, 1.0, 5.0))

    # ----- orders (with a few planted anomalies for Assignment 4) -----------
    orders = []
    for oid in range(4000):
        cust = customers[int(rng.integers(0, len(customers)))]
        # customers mostly order food they like: sample 3 dishes, keep best
        cands = [dishes[int(rng.integers(0, len(dishes)))] for _ in range(3)]
        dish = max(cands, key=lambda d: affinity(cust, d) + rng.normal(0, 0.7))
        true_score = affinity(cust, dish)
        rating = int(np.clip(round(true_score + rng.normal(0, 0.55)), 1, 5))
        delivery_min = float(np.clip(rng.normal(37, 9), 12, 95))
        tip = float(np.round(max(0.0, (rating - 2.2) * 1.3 + rng.normal(0, 0.7)), 2))
        is_anomaly = False
        if oid % 50 == 7:                     # 2% of orders are "weird"
            is_anomaly = True
            kind = oid % 3
            if kind == 0:
                delivery_min, tip = 2.0, 45.0            # impossible + huge tip
            elif kind == 1:
                delivery_min, rating = 240.0, 5          # 4-hour "happy" order
            else:
                tip, rating = 38.0, 1                    # giant tip, 1 star
        orders.append({
            "order_id": oid,
            "customer_id": cust["customer_id"],
            "dish_id": dish["dish_id"],
            "rating": rating,                 # 1..5 stars
            "delivery_min": round(delivery_min, 1),
            "tip": tip,
            "reordered": bool(true_score + rng.normal(0, 0.45) > 3.55),
            "_is_anomaly": is_anomaly,        # hidden ground truth (A4 checks)
        })

    # ----- written reviews (roughly 1 in 3 orders gets one) ------------------
    reviews = []
    for od in orders:
        if od["order_id"] % 3 != 0:
            continue
        dish = dishes[od["dish_id"]]
        tmpl = _REVIEW_TEMPLATES[od["rating"]][od["order_id"] % 4]
        ing_ids = np.flatnonzero(dish["ingredients"])
        ing = INGREDIENTS[int(ing_ids[od["order_id"] % len(ing_ids)])]
        text = tmpl.format(dish=dish["name"].lower(), ing=ing)
        if dish["spice"] >= 2:
            text += _SPICY_TAIL.format(
                level="fiery" if dish["spice"] == 3 else "warm",
                verdict="perfect" if od["rating"] >= 4 else "too much")
        reviews.append({"order_id": od["order_id"], "rating": od["rating"],
                        "text": text})

    # ----- dish photos: tiny 16x16 grayscale images, 3 classes ---------------
    # class 0 = pizza (a filled circle), class 1 = sushi (a grid of squares),
    # class 2 = noodles (wavy horizontal stripes). Real enough for a real CNN!
    def _draw(kind, r):
        img = np.zeros((16, 16), dtype=np.float32)
        yy, xx = np.mgrid[0:16, 0:16]
        if kind == 0:      # pizza: disc + darker "pepperoni" dots
            cy, cx = 7.5 + r.normal(0, 0.7), 7.5 + r.normal(0, 0.7)
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = 0.85
            for _ in range(4):
                py, px = int(r.integers(4, 12)), int(r.integers(4, 12))
                img[max(py-1, 0):py+1, max(px-1, 0):px+1] = 0.25
        elif kind == 1:    # sushi: 2x2 grid of bright squares
            off = int(r.integers(0, 2))
            for gy in (2 + off, 9 + off):
                for gx in (2 + off, 9 + off):
                    img[gy:gy+5, gx:gx+5] = 0.9
        else:              # noodles: 3 wavy stripes
            phase = r.uniform(0, 3.14)
            for row in (3, 8, 13):
                wave = (row + 1.6 * np.sin(xx[0] / 2.1 + phase)).astype(int)
                for c in range(16):
                    rr = int(np.clip(wave[c] + r.integers(-1, 2), 0, 15))
                    img[rr, c] = 0.8
        img += r.normal(0, 0.13, size=(16, 16)).astype(np.float32)
        return np.clip(img, 0.0, 1.0)

    img_rng = np.random.default_rng(SEED + 7)
    images, image_labels = [], []
    for kind in (0, 1, 2):
        for _ in range(200):
            images.append(_draw(kind, img_rng))
            image_labels.append(kind)
    perm = img_rng.permutation(len(images))
    images = np.stack(images)[perm]
    image_labels = np.array(image_labels)[perm]
    IMAGE_CLASSES = ["pizza", "sushi", "noodles"]

    return {
        "customers": customers, "dishes": dishes, "orders": orders,
        "reviews": reviews, "faq_docs": FAQ_DOCS,
        "images": images, "image_labels": image_labels,
        "image_classes": IMAGE_CLASSES,
    }


_WORLD = None

def world():
    """The company dataset (built once, then cached)."""
    global _WORLD
    if _WORLD is None:
        _WORLD = build_world()
    return _WORLD


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 1 — "MEET THE DATA"
#   Course topic: Introduction to deep learning (Weeks 1-2)
#   Difficulty: ▉░░░░░░░░░        Time: ~half a week
#
# ==========================================================================
# THE STORY
#   Your boss, on your first morning: "Welcome aboard! Before you build
#   anything fancy, get to know our data. Then show me one tiny neural
#   network that learns SOMETHING — anything — so I know this ML stuff
#   actually works."
#
# YOUR GOAL
#   1. Answer three basic questions about the company data.
#   2. Watch a tiny neural network learn to predict whether a customer
#      will reorder a dish — and beat random guessing.
#
# WHAT YOU'LL LEARN
#   What the data looks like, what a "feature" is, and what one full
#   training loop of a neural network actually does, line by line.
# ==========================================================================

# --- PAUSE AND THINK 1.5 (fill this in when the assignment asks you to) ---
THINK_1_5 = """
"""  # TODO[1.5]: after training runs, explain IN YOUR OWN WORDS why the
     # loss number goes DOWN as the network trains. What is the network
     # "doing" between epoch 1 and epoch 10? (2-3 sentences here.)


def _order_features(od, w):
    """Turn one order into a list of plain numbers — a 'feature vector'.

    Neural networks can't read words like 'Pad Thai' or 'thai'. They eat
    fixed-length lists of numbers. This function is that translation step,
    and EVERY later assignment builds on this idea.
    """
    cust = w["customers"][od["customer_id"]]
    dish = w["dishes"][od["dish_id"]]
    return [
        dish["price"],                                        # how expensive
        float(dish["spice"]),                                 # how spicy 0-3
        float(cust["spice_tolerance"]),                       # how brave 0-3
        1.0 if dish["cuisine"] == cust["favorite_cuisine"] else 0.0,
        float(cust["budget"]),
        od["delivery_min"],
    ]


def assignment_1():
    banner("ASSIGNMENT 1 — MEET THE DATA")
    w = world()

    # ----------------------------------------------------------------------
    # STEP 1.1 — How big is this company, anyway?
    #   The data lives in three Python lists:
    #       w["customers"]   w["dishes"]   w["orders"]
    #   The len() function counts items in a list:  len([10, 20, 30]) -> 3
    # ----------------------------------------------------------------------
    say("", "STEP 1.1 — count what we have")

    n_customers = None   # TODO[1.1a]: replace None with  len(w["customers"])
    n_dishes    = None   # TODO[1.1b]: count the dishes the same way
    n_orders    = None   # TODO[1.1c]: and the orders

    need_value(n_customers, "1.1a", "Use len() on the customers list.")
    need_value(n_dishes,    "1.1b", "Same as 1.1a, but for w['dishes'].")
    need_value(n_orders,    "1.1c", "Same idea, for w['orders'].")
    checkpoint("1.1", (n_customers, n_dishes, n_orders) == (200, 60, 4000),
               f"{n_customers} customers, {n_dishes} dishes, {n_orders} orders.",
               "Each answer should use len() on the matching list — check "
               "you didn't mix up which list goes with which variable.")

    # ----------------------------------------------------------------------
    # STEP 1.2 — What's our average rating?
    #   Every order has a star rating from 1 to 5, stored as od["rating"].
    #   We've collected them into a plain list of numbers for you:
    # ----------------------------------------------------------------------
    say("", "STEP 1.2 — the average star rating")
    all_ratings = [od["rating"] for od in w["orders"]]   # e.g. [4, 5, 2, ...]

    avg_rating = None   # TODO[1.2]: the average = sum(all_ratings) / len(all_ratings)

    need_value(avg_rating, "1.2", "sum(...) divided by len(...) — one line.")
    checkpoint("1.2", 3.3 < avg_rating < 3.9,
               f"average rating = {avg_rating:.2f} stars. Customers are fairly happy.",
               "Expected a number between 3.3 and 3.9. Did you divide by the "
               "length of the SAME list you summed?")

    # ----------------------------------------------------------------------
    # STEP 1.3 — Which cuisine sells the most?
    #   We've built you a dictionary of counts, like:
    #       {"thai": 812, "italian": 640, ...}
    #   Python's max() can find the key with the biggest value:
    #       max(counts, key=counts.get)
    # ----------------------------------------------------------------------
    say("", "STEP 1.3 — the best-selling cuisine")
    counts = {}
    for od in w["orders"]:
        cuisine = w["dishes"][od["dish_id"]]["cuisine"]
        counts[cuisine] = counts.get(cuisine, 0) + 1
    say("   orders per cuisine: " +
        ", ".join(f"{c}={counts[c]}" for c in sorted(counts, key=counts.get, reverse=True)))

    top_cuisine = None   # TODO[1.3]: use max(counts, key=counts.get)

    need_value(top_cuisine, "1.3", "Copy the max(...) pattern from the comment above.")
    checkpoint("1.3", top_cuisine == max(counts, key=counts.get),
               f"'{top_cuisine}' is our best seller.",
               "max(counts, key=counts.get) returns the cuisine name with "
               "the highest count.")

    # ----------------------------------------------------------------------
    # STEP 1.4 — From order to numbers (the most important idea in ML2!)
    #   Scroll up and READ the function _order_features(). It turns one
    #   order into 6 numbers. Below, we print one example so you can see it.
    #   Your job: just say how many numbers ("features") each order becomes.
    # ----------------------------------------------------------------------
    say("", "STEP 1.4 — every order becomes a list of numbers")
    example = _order_features(w["orders"][0], w)
    say(f"   order #0 as numbers: {[round(x, 1) for x in example]}")

    n_features = None   # TODO[1.4]: how long is that list? (count, or use len(example))

    need_value(n_features, "1.4", "It's the length of the example list printed above.")
    checkpoint("1.4", n_features == 6,
               "6 features per order. The network below has 6 'input wires'.",
               "Count the entries in _order_features()'s returned list, or "
               "just use len(example).")

    # ----------------------------------------------------------------------
    # STEP 1.5 — Your first neural network
    #   The question it learns to answer: "will this customer REORDER this
    #   dish?" (True/False — stored in od["reordered"]).
    #
    #   The training loop below is THE loop of all deep learning. The same
    #   5 beats repeat in Assignments 2, 4, 6, 7, and 8:
    #      1. predict     (model makes a guess)
    #      2. measure     (loss = how wrong the guess was)
    #      3. reset       (clear old gradients)
    #      4. backward    (work out which weights caused the error)
    #      5. step        (nudge every weight to be slightly less wrong)
    #   You only choose how many EPOCHS (full passes over the data) to run.
    # ----------------------------------------------------------------------
    say("", "STEP 1.5 — train the tiny network")

    EPOCHS = None   # TODO[1.5e]: start with 10

    need_value(EPOCHS, "1.5e", "Set EPOCHS = 10 to begin with.")

    torch.manual_seed(SEED)
    X = torch.tensor([_order_features(od, w) for od in w["orders"]])
    y = torch.tensor([1.0 if od["reordered"] else 0.0 for od in w["orders"]])
    X = (X - X.mean(0)) / X.std(0)          # scale features (explained in A2!)

    model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()        # loss for yes/no questions

    for epoch in range(EPOCHS):
        pred = model(X).squeeze()           # 1. predict
        loss = loss_fn(pred, y)             # 2. measure
        optimizer.zero_grad()               # 3. reset
        loss.backward()                     # 4. backward
        optimizer.step()                    # 5. step
        if epoch % max(1, EPOCHS // 10) == 0 or epoch == EPOCHS - 1:
            say(f"   epoch {epoch:>3}   loss = {loss.item():.3f}")

    with torch.no_grad():
        accuracy = ((model(X).squeeze() > 0) == (y > 0.5)).float().mean().item()
    base_rate = max(y.mean().item(), 1 - y.mean().item())
    say(f"   accuracy = {accuracy:.1%}   (always guessing the majority would get {base_rate:.1%})")

    checkpoint("1.5", accuracy > base_rate + 0.03,
               f"your network beats blind guessing by {accuracy - base_rate:+.1%}. It LEARNED.",
               "Accuracy should clearly beat the majority-guess rate. Is "
               "EPOCHS at least 10? Try 30.")

    think_check("1.5", THINK_1_5)

    ARTIFACTS[1] = {"order_features": _order_features}
    _mark_done(1)
    say("", "🎉 ASSIGNMENT 1 COMPLETE — your boss is mildly impressed.",
        "   Next:  python ml2_final_project.py 2")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 2 — "PREDICT DELIVERY RATINGS"
#   Course topic: Building a real-world predictor (Week 3)
#   Difficulty: ▉▉░░░░░░░░        Time: ~1 week
#
# ==========================================================================
# THE STORY
#   Boss: "Support keeps getting surprise 1-star ratings. If we could
#   PREDICT a bad rating before the customer leaves it, we could send an
#   apology coupon first. Can you predict the star rating of an order?"
#
# YOUR GOAL
#   A properly-built regression model: honest train/validation/test split,
#   scaled features, a training loop YOU write the heart of, and a hard
#   look at overfitting — the classic failure of real ML projects.
#
# WHAT YOU'LL LEARN
#   Why we split data three ways, why features get scaled, the three
#   sacred lines of every PyTorch loop, and what overfitting looks like
#   on a chart you made yourself.
# ==========================================================================

THINK_2_5 = """
"""  # TODO[2.5t]: The big model scored BETTER on training data but WORSE on
     # validation data than the small model. In your own words: what
     # happened, and why is the validation number the one we should trust?


def assignment_2():
    banner("ASSIGNMENT 2 — PREDICT DELIVERY RATINGS")
    w = world()
    a1 = need(1, assignment_1)
    feats = a1["order_features"]

    X_all = np.array([feats(od, w) for od in w["orders"]], dtype=np.float32)
    y_all = np.array([od["rating"] for od in w["orders"]], dtype=np.float32)

    # ----------------------------------------------------------------------
    # STEP 2.1 — Split the data (never grade yourself on your own homework)
    #   TRAIN      = the model learns from this           (first 2800 orders)
    #   VALIDATION = we tune decisions using this          (next 600)
    #   TEST       = touched ONCE at the very end          (last 600)
    #   Python slicing:  X_all[:2800] takes the first 2800 rows,
    #                    X_all[2800:3400] takes rows 2800..3399, etc.
    # ----------------------------------------------------------------------
    say("", "STEP 2.1 — split 4000 orders into train / validation / test")

    X_train, y_train = X_all[:2800], y_all[:2800]
    X_val   = None   # TODO[2.1a]: rows 2800 to 3400 of X_all  ->  X_all[2800:3400]
    y_val   = None   # TODO[2.1b]: same rows of y_all
    X_test  = None   # TODO[2.1c]: everything from row 3400 on ->  X_all[3400:]
    y_test  = None   # TODO[2.1d]: same rows of y_all

    need_value(X_val,  "2.1a", "X_all[2800:3400]")
    need_value(y_val,  "2.1b", "y_all[2800:3400]")
    need_value(X_test, "2.1c", "X_all[3400:]")
    need_value(y_test, "2.1d", "y_all[3400:]")
    checkpoint("2.1",
               len(X_val) == 600 and len(X_test) == 600 and len(y_val) == 600
               and len(y_test) == 600,
               "2800 train / 600 validation / 600 test. Honest bookkeeping.",
               "Check your slice numbers: [2800:3400] gives 600 rows, "
               "[3400:] gives the last 600.")

    # ----------------------------------------------------------------------
    # STEP 2.2 — Scale the features
    #   'price' runs 8-24 but 'match' is 0-1. Unscaled, the network wastes
    #   ages compensating for the big numbers. The classic fix: for each
    #   feature, subtract the mean and divide by the standard deviation —
    #   then every feature hovers around 0 with a spread of about 1.
    #   CRITICAL RULE: compute mean/std on TRAIN ONLY (the model must not
    #   peek at validation or test data, even indirectly).
    # ----------------------------------------------------------------------
    say("", "STEP 2.2 — put all features on a comparable scale")
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mu) / std
    X_val   = None   # TODO[2.2a]: scale X_val the same way: (X_val - mu) / std
    X_test  = None   # TODO[2.2b]: and X_test — SAME mu and std, from train!

    need_value(X_val,  "2.2a", "(X_val - mu) / std  — reuse train's mu and std.")
    need_value(X_test, "2.2b", "(X_test - mu) / std")
    checkpoint("2.2", abs(float(X_train.mean())) < 0.01 and X_val.shape == (600, 6),
               "features scaled — every column now lives near 0.",
               "Use the mu and std computed above (from train), don't "
               "recompute them on val/test.")

    # ----------------------------------------------------------------------
    # STEP 2.3 — Build the model & pick the loss
    #   Ratings are numbers (1..5), so this is REGRESSION. The standard loss
    #   is Mean Squared Error: the average of (guess - truth)² over a batch.
    #   In plain English: big mistakes get punished much harder than small
    #   ones, and 0 loss means every guess was perfect.
    # ----------------------------------------------------------------------
    say("", "STEP 2.3 — model and loss")

    HIDDEN = None    # TODO[2.3a]: width of the hidden layer — start with 16

    need_value(HIDDEN, "2.3a", "HIDDEN = 16 is a sensible start.")

    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(6, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    loss_fn = None   # TODO[2.3b]: nn.MSELoss()   <- mean squared error

    need_value(loss_fn, "2.3b", "loss_fn = nn.MSELoss()")
    checkpoint("2.3", isinstance(loss_fn, nn.MSELoss),
               f"model built (6 -> {HIDDEN} -> 1) with MSE loss.",
               "loss_fn must be an instance: nn.MSELoss() with parentheses.")

    # ----------------------------------------------------------------------
    # STEP 2.4 — Write the heart of the training loop yourself
    #   In A1 we gave you the 5 beats. This time YOU write beats 3-5.
    #   They are, in order (exactly one line each):
    #       optimizer.zero_grad()     <- reset:    clear old gradients
    #       loss.backward()           <- backward: assign blame to weights
    #       optimizer.step()          <- step:     nudge weights downhill
    #   These three lines appear in every PyTorch project on Earth. After
    #   this assignment you'll never look them up again.
    # ----------------------------------------------------------------------
    say("", "STEP 2.4 — train (you write the famous three lines)")

    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train_one_epoch():
        pred = model(Xtr).squeeze()          # 1. predict
        loss = loss_fn(pred, ytr)            # 2. measure
        # ===== TODO[2.4] BEGIN: write the three lines (reset, backward, step)
        raise NotImplementedError("TODO 2.4")
        # ===== TODO[2.4] END =====
        return loss.item()

    try:
        first_loss = train_one_epoch()
    except NotImplementedError:
        say("",
            "  ⏸  TODO[2.4] is still waiting: replace the raise line with the",
            "     three lines listed in the comment above (keep their order!).")
        raise NotDoneYet("2.4")

    for epoch in range(1, 60):
        last_loss = train_one_epoch()
        if epoch % 10 == 0:
            say(f"   epoch {epoch:>3}   train loss = {last_loss:.3f}")

    checkpoint("2.4", last_loss < first_loss * 0.8,
               f"loss fell {first_loss:.2f} -> {last_loss:.2f}. The loop works.",
               "Loss should drop clearly. The three lines must be in order: "
               "zero_grad, backward, step — and inside the function, before "
               "'return'.")

    def mae(m, X, y):
        """Mean Absolute Error: on average, how many stars off are we?"""
        with torch.no_grad():
            p = m(torch.tensor(X)).squeeze().numpy()
        return float(np.abs(p - y).mean())

    say(f"   -> average error: train {mae(model, X_train, y_train):.2f} stars, "
        f"validation {mae(model, X_val, y_val):.2f} stars")

    # ----------------------------------------------------------------------
    # STEP 2.5 — The overfitting experiment
    #   We now train a much BIGGER network for much LONGER on the SAME data,
    #   and compare. Watch what happens to train error vs validation error.
    # ----------------------------------------------------------------------
    say("", "STEP 2.5 — the overfitting experiment (no code to write — watch!)")
    torch.manual_seed(SEED)
    big = nn.Sequential(nn.Linear(6, 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, 1))
    big_opt = torch.optim.Adam(big.parameters(), lr=0.01)
    for epoch in range(700):
        pred = big(Xtr).squeeze()
        loss = loss_fn(pred, ytr)
        big_opt.zero_grad(); loss.backward(); big_opt.step()

    small_tr, small_val = mae(model, X_train, y_train), mae(model, X_val, y_val)
    big_tr,   big_val   = mae(big, X_train, y_train),   mae(big, X_val, y_val)
    say("",
        "                      train error   validation error",
        f"   small model (yours)    {small_tr:.3f}          {small_val:.3f}",
        f"   big model (700 ep)     {big_tr:.3f}          {big_val:.3f}",
        "",
        "   The big model looks like a genius on data it has seen, and does",
        "   no better (or worse) on data it hasn't. It has started MEMORIZING",
        "   instead of LEARNING. That is overfitting.")

    think_check("2.5t", THINK_2_5)

    # Final honest number — the test set, touched exactly once:
    test_mae = mae(model, X_test, y_test)
    say(f"   FINAL (test set, used once): your model is off by {test_mae:.2f} stars on average.")
    checkpoint("2.5", test_mae < 0.85,
               f"test MAE = {test_mae:.2f} — good enough to catch unhappy customers.",
               "Test error should be under 0.85 stars. If not, check HIDDEN "
               "= 16 and that step 2.4's three lines are right.")

    ARTIFACTS[2] = {"rating_model": model, "mu": mu, "std": std, "mae": mae}
    _mark_done(2)
    say("", "🎉 ASSIGNMENT 2 COMPLETE — support can now pre-empt bad reviews.",
        "   Next:  python ml2_final_project.py 3")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 3 — "CUSTOMERS WHO LIKE X ALSO LIKE Y"
#   Course topic: Vector representations & similarity (Week 4)
#   Difficulty: ▉▉▉░░░░░░░        Time: ~1 week
#
# ==========================================================================
# THE STORY
#   Boss: "Customers keep ordering the same two dishes forever, then get
#   bored and quit the app. Suggest dishes they'd probably like!"
#
# YOUR GOAL
#   A working recommend(customer) function that returns 3 dishes the
#   customer hasn't tried, ranked by how well they match their taste.
#
# THE BIG IDEA
#   A computer can't compare "Pad Thai" with "Lasagna". But if each dish
#   becomes a VECTOR (a list of numbers describing it), then "similar
#   dishes" becomes "vectors pointing in a similar direction" — and THAT
#   we can measure, with one formula: cosine similarity.
# ==========================================================================

THINK_3_5 = """
"""  # TODO[3.5]: your recommender suggested dishes without ever being told
     # what "tastes similar" means. Where, exactly, did that knowledge come
     # from? (Think: what did we average, and what does that vector contain?)


def _dish_vector(dish):
    """A dish as 32 numbers: 24 ingredients + 6 cuisine slots + price + spice."""
    vec = np.zeros(32, dtype=np.float32)
    vec[:24] = dish["ingredients"]                      # which ingredients
    # one-hot cuisine: exactly one of positions 24..29 becomes 1.0
    vec[24 + CUISINES.index(dish["cuisine"])] = 2.0     # cuisine (weighted x2)
    vec[30] = dish["price"] / 24.0                      # price, squashed to ~0..1
    vec[31] = dish["spice"] / 3.0                       # spice, squashed to 0..1
    return vec


def assignment_3():
    banner("ASSIGNMENT 3 — CUSTOMERS WHO LIKE X ALSO LIKE Y")
    w = world()

    # ----------------------------------------------------------------------
    # STEP 3.1 — Look at dishes as vectors
    #   Read _dish_vector() above — it's already written. Below we print two
    #   similar dishes and one very different one. Just answer: which TWO of
    #   the three printed dishes should have the most similar vectors?
    # ----------------------------------------------------------------------
    say("", "STEP 3.1 — dishes are now arrows in 32-dimensional space")
    d_pad, d_drunken, d_burger = w["dishes"][40], w["dishes"][48], w["dishes"][50]
    for d in (d_pad, d_drunken, d_burger):
        say(f"   {d['name']:<22} cuisine={d['cuisine']:<9} spice={d['spice']}  "
            f"price=${d['price']:.2f}")

    # Which pair is most similar? Answer with two of: "pad", "drunken", "burger"
    similar_pair = (None, None)   # TODO[3.1]: e.g. ("pad", "burger") — pick the right two!

    need_value(similar_pair[0], "3.1", 'Fill both slots, e.g. ("pad", "drunken").')
    checkpoint("3.1", set(similar_pair) == {"pad", "drunken"},
               "yes — two thai noodle dishes beat a burger every time.",
               "Two of these dishes share cuisine, ingredients and spice. "
               "The burger isn't one of them.")

    # ----------------------------------------------------------------------
    # STEP 3.2 — Write cosine similarity (your first real formula!)
    #   cosine(a, b) = (a · b) / (|a| * |b|)
    #   In plain English: multiply matching entries and add them up (the dot
    #   product rewards shared ingredients), then divide by both lengths so
    #   a big expensive dish can't win just by having bigger numbers.
    #   Result: +1 = same direction (twins), 0 = unrelated.
    #   Your numpy toolbox:   np.dot(a, b)         the dot product
    #                         np.linalg.norm(a)    the length |a|
    # ----------------------------------------------------------------------
    say("", "STEP 3.2 — the similarity formula")

    def cosine_sim(a, b):
        # ===== TODO[3.2] BEGIN: return the formula above (1-3 lines)
        raise NotImplementedError("TODO 3.2")
        # ===== TODO[3.2] END =====

    try:
        test_val = cosine_sim(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    except NotImplementedError:
        say("", "  ⏸  TODO[3.2]: replace the raise line with the cosine formula.",
            "     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))")
        raise NotDoneYet("3.2")

    same = cosine_sim(np.array([2.0, 2.0]), np.array([4.0, 4.0]))
    perp = cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    checkpoint("3.2", abs(test_val - 1) < 1e-5 and abs(same - 1) < 1e-5 and abs(perp) < 1e-5,
               "cosine_sim works: twins score 1.0, unrelated score 0.0.",
               "Parallel vectors (even different lengths) must give exactly "
               "1.0, perpendicular ones 0.0. Check the formula's parentheses.")

    sim_noodles = cosine_sim(_dish_vector(d_pad), _dish_vector(d_drunken))
    sim_burger  = cosine_sim(_dish_vector(d_pad), _dish_vector(d_burger))
    say(f"   Pad Thai vs Drunken Noodles: {sim_noodles:.2f}   "
        f"Pad Thai vs Cheeseburger: {sim_burger:.2f}   <- the math agrees with you")

    # ----------------------------------------------------------------------
    # STEP 3.3 — A customer's "taste vector"
    #   Idea: a customer IS the average of the dishes they loved.
    #   We collect the vectors of every dish this customer rated 4+ stars,
    #   then average them into one vector that points at their taste.
    # ----------------------------------------------------------------------
    say("", "STEP 3.3 — turn a customer into a vector too")

    LIKED_MIN_RATING = None   # TODO[3.3]: how many stars counts as "liked"? Use 4

    need_value(LIKED_MIN_RATING, "3.3", "LIKED_MIN_RATING = 4")

    def taste_vector(customer_id):
        liked = [_dish_vector(w["dishes"][od["dish_id"]])
                 for od in w["orders"]
                 if od["customer_id"] == customer_id
                 and od["rating"] >= LIKED_MIN_RATING]
        if not liked:
            return None                      # brand-new customer: no history
        return np.mean(liked, axis=0)

    tv = taste_vector(7)
    checkpoint("3.3", LIKED_MIN_RATING == 4 and tv is not None and tv.shape == (32,),
               "customer #7 is now a 32-number taste vector.",
               "Use 4 as the threshold — 3 stars means 'meh', not 'liked'.")

    # ----------------------------------------------------------------------
    # STEP 3.4 — Recommend!
    #   For one customer: score every dish they HAVEN'T ordered by cosine
    #   similarity to their taste vector, then return the 3 best.
    #   Your toolbox:  np.argsort(scores)  gives positions sorted LOW to HIGH
    #                  [::-1]              reverses a list (now HIGH to LOW)
    #                  [:3]                keeps the first 3
    # ----------------------------------------------------------------------
    say("", "STEP 3.4 — the recommendation function")

    def recommend(customer_id):
        taste = taste_vector(customer_id)
        if taste is None:
            return []
        tried = {od["dish_id"] for od in w["orders"]
                 if od["customer_id"] == customer_id}
        candidates = [d for d in w["dishes"] if d["dish_id"] not in tried]
        scores = np.array([cosine_sim(taste, _dish_vector(d)) for d in candidates])

        best3 = None   # TODO[3.4]: positions of the 3 highest scores
                       #            pattern: np.argsort(scores)[::-1][:3]

        need_value(best3, "3.4", "np.argsort(scores)[::-1][:3]")
        return [candidates[i] for i in best3]

    recs = recommend(7)
    cust7 = w["customers"][7]
    say(f"   {cust7['name']} (loves {cust7['favorite_cuisine']}, spice tolerance "
        f"{cust7['spice_tolerance']}) should try:")
    for d in recs:
        say(f"      -> {d['name']}  ({d['cuisine']}, spice {d['spice']})")

    # Grade the recommender across 30 customers: recommendations should
    # match each customer's favorite cuisine far more often than chance.
    hits = total = 0
    for cid in range(30):
        for d in recommend(cid):
            total += 1
            hits += (d["cuisine"] == w["customers"][cid]["favorite_cuisine"])
    rate = hits / max(total, 1)
    say(f"   across 30 customers: {rate:.0%} of recommendations match the "
        f"customer's favorite cuisine (random would be ~17%)")
    checkpoint("3.4", rate > 0.45,
               "the recommender clearly tracks personal taste. Ship it!",
               "Should be well above 45%. Check best3 keeps the HIGHEST "
               "scores (did you forget [::-1]?).")

    # ----------------------------------------------------------------------
    # STEP 3.5 — Pause and think
    # ----------------------------------------------------------------------
    think_check("3.5", THINK_3_5)

    ARTIFACTS[3] = {"cosine_sim": cosine_sim, "dish_vector": _dish_vector,
                    "taste_vector": taste_vector, "recommend": recommend}
    _mark_done(3)
    say("", "🎉 ASSIGNMENT 3 COMPLETE — the app now has a 'For You' tab.",
        "   Next:  python ml2_final_project.py 4")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 4 — "FIND THE WEIRD ORDERS"
#   Course topic: Autoencoders (Week 5)
#   Difficulty: ▉▉▉▉░░░░░░        Time: ~1 week
#
# ==========================================================================
# THE STORY
#   Boss: "Finance flagged some orders that make no sense — 2-minute
#   deliveries, $45 tips on 1-star meals. We think there's fraud or a bug.
#   Nobody has time to eyeball 4000 orders. Find the weird ones."
#
# YOUR GOAL
#   Train an AUTOENCODER — a network that learns to squeeze each order
#   through a tiny bottleneck and rebuild it. It gets good at rebuilding
#   NORMAL orders. Weird orders come out mangled... and the size of the
#   mangling is your fraud alarm.
#
# WHAT YOU'LL LEARN
#   That a network trained with NO labels at all can still find structure —
#   and that "how badly did reconstruction fail" is a powerful signal.
# ==========================================================================

THINK_4_5 = """
"""  # TODO[4.5]: we never told the autoencoder what "weird" means, and yet
     # it found the weird orders. Explain in your own words why unusual
     # orders get bigger reconstruction errors than typical ones.


def assignment_4():
    banner("ASSIGNMENT 4 — FIND THE WEIRD ORDERS")
    w = world()

    # Features chosen to expose weirdness: money and time, mostly.
    X = np.array([[od["delivery_min"], od["tip"], float(od["rating"]),
                   w["dishes"][od["dish_id"]]["price"]]
                  for od in w["orders"]], dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)          # scale, like A2 taught us
    Xt = torch.tensor(X)

    # ----------------------------------------------------------------------
    # STEP 4.1 — Design the bottleneck
    #   Our orders are 4 numbers wide. The autoencoder squeezes them through
    #   a bottleneck of size B, then tries to rebuild all 4 numbers from
    #   just those B. For the trick to work, B must be SMALLER than 4 —
    #   if B=4 the network can just copy its input and learns nothing.
    # ----------------------------------------------------------------------
    say("", "STEP 4.1 — how tight should the squeeze be?")

    BOTTLENECK = None   # TODO[4.1]: pick a size SMALLER than 4. Try 2.

    need_value(BOTTLENECK, "4.1", "BOTTLENECK = 2 works well.")
    checkpoint("4.1", 1 <= BOTTLENECK <= 3,
               f"squeezing 4 numbers down to {BOTTLENECK} and back.",
               "Must be 1, 2 or 3 — a bottleneck of 4+ can cheat by copying.")

    torch.manual_seed(SEED)
    autoencoder = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, BOTTLENECK),   # encoder
        nn.Linear(BOTTLENECK, 8), nn.ReLU(), nn.Linear(8, 4),   # decoder
    )

    # ----------------------------------------------------------------------
    # STEP 4.2 — Train it (the three sacred lines return!)
    #   Note what's DIFFERENT from A2: the target is not a label — it's the
    #   INPUT ITSELF. loss = MSE(rebuilt, original). No labels anywhere.
    # ----------------------------------------------------------------------
    say("", "STEP 4.2 — train the autoencoder")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    def train_one_epoch():
        rebuilt = autoencoder(Xt)                # 1. predict (rebuild input)
        loss = loss_fn(rebuilt, Xt)              # 2. measure vs the INPUT
        # ===== TODO[4.2] BEGIN: the same three lines you wrote in A2
        raise NotImplementedError("TODO 4.2")
        # ===== TODO[4.2] END =====
        return loss.item()

    try:
        first_loss = train_one_epoch()
    except NotImplementedError:
        say("", "  ⏸  TODO[4.2]: same three lines as Assignment 2, step 2.4.",
            "     (zero_grad, backward, step — you know these now.)")
        raise NotDoneYet("4.2")

    for epoch in range(1, 300):
        last_loss = train_one_epoch()
        if epoch % 75 == 0:
            say(f"   epoch {epoch:>3}   rebuild loss = {last_loss:.4f}")
    checkpoint("4.2", last_loss < first_loss * 0.5,
               f"rebuild loss fell {first_loss:.3f} -> {last_loss:.3f}.",
               "Loss should at least halve. Are the three lines in order and "
               "before 'return'?")

    # ----------------------------------------------------------------------
    # STEP 4.3 — Score every order by reconstruction error
    #   For each order: how far is the rebuilt version from the original?
    #   error = mean of (original - rebuilt)²  across the 4 features.
    #   With numpy that's one line:  ((X - rebuilt)**2).mean(axis=1)
    #   (axis=1 means "average across each ROW", giving one score per order.)
    # ----------------------------------------------------------------------
    say("", "STEP 4.3 — one weirdness score per order")
    with torch.no_grad():
        rebuilt = autoencoder(Xt).numpy()

    errors = None   # TODO[4.3]: ((X - rebuilt) ** 2).mean(axis=1)

    need_value(errors, "4.3", "((X - rebuilt) ** 2).mean(axis=1)")
    checkpoint("4.3", errors.shape == (4000,) and float(errors.min()) >= 0,
               "4000 weirdness scores computed.",
               "Result must have exactly one number per order: shape (4000,). "
               "Did you use axis=1?")

    # ----------------------------------------------------------------------
    # STEP 4.4 — Sound the alarm on the top 2%
    #   np.percentile(errors, 98) gives the score that 98% of orders sit
    #   below. Anything above that line gets flagged.
    # ----------------------------------------------------------------------
    say("", "STEP 4.4 — flag the most suspicious 2%")

    threshold = None   # TODO[4.4]: np.percentile(errors, 98)

    need_value(threshold, "4.4", "np.percentile(errors, 98)")
    flagged = np.flatnonzero(errors > threshold)
    say(f"   flagged {len(flagged)} orders. A few of them:")
    for oid in flagged[:5]:
        od = w["orders"][int(oid)]
        say(f"      order #{od['order_id']:<5} delivery {od['delivery_min']:>6.1f} min,"
            f" tip ${od['tip']:>5.2f}, rating {od['rating']}")

    # Grade against the hidden ground truth (finance's secret list):
    truly_weird = {od["order_id"] for od in w["orders"] if od["_is_anomaly"]}
    caught = len(truly_weird.intersection(int(i) for i in flagged))
    say(f"   ...finance confirms: {caught} of the {len(truly_weird)} truly "
        f"weird orders are in your flagged list.")
    checkpoint("4.4", caught >= len(truly_weird) * 0.45,
               f"you caught {caught}/{len(truly_weird)} with zero labels. Magic? No — autoencoders.",
               "You should catch close to half. Check BOTTLENECK is 2 (not "
               "3), and that 4.3 used axis=1.")

    # ----------------------------------------------------------------------
    # STEP 4.5 — Pause and think
    # ----------------------------------------------------------------------
    think_check("4.5", THINK_4_5)

    ARTIFACTS[4] = {"autoencoder": autoencoder, "errors": errors}
    _mark_done(4)
    say("", "🎉 ASSIGNMENT 4 COMPLETE — finance owes you a coffee.",
        "   Next:  python ml2_final_project.py 5")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 5 — "WHAT DOES 'SPICY' MEAN TO A COMPUTER?"
#   Course topic: Word embeddings (Week 6)
#   Difficulty: ▉▉▉▉▉░░░░░        Time: ~1 week
#
# ==========================================================================
# THE STORY
#   Boss: "Marketing wants to auto-tag reviews: which words mean happy,
#   which mean angry, which are about food vs delivery. Can the computer
#   learn what our customers' words MEAN?"
#
# YOUR GOAL
#   Build word vectors from FlavorCart's own reviews — using the idea
#   behind Word2Vec: "a word is known by the company it keeps." Words that
#   appear in similar surroundings get similar vectors.
#
# HOW (in plain English)
#   1. Count, for every pair of words, how often they appear near each
#      other  ->  a big table of counts.
#   2. Turn raw counts into "appears together MORE than chance would
#      predict" scores (that's all PPMI means).
#   3. Compress each word's row of scores down to 32 numbers (like the
#      autoencoder's bottleneck — same idea, done with matrix math).
#   The result: every word is a vector, and cosine similarity (yours,
#   from A3!) measures similarity in MEANING.
# ==========================================================================

THINK_5_5 = """
"""  # TODO[5.5]: the computer never saw a dictionary, yet "cold" and
     # "soggy" ended up neighbors. Explain in your own words how ONLY
     # counting nearby words made that happen.


def assignment_5():
    banner("ASSIGNMENT 5 — WHAT DOES 'SPICY' MEAN TO A COMPUTER?")
    w = world()
    a3 = need(3, assignment_3)
    cosine_sim = a3["cosine_sim"]

    # ----------------------------------------------------------------------
    # STEP 5.1 — Chop reviews into words ("tokenization", the simple way)
    #   Real LLMs use fancy sub-word tokenizers (Assignment 8 revisits
    #   this). Today, lowercase + split on spaces is honest work.
    # ----------------------------------------------------------------------
    say("", "STEP 5.1 — tokenize the reviews")
    say(f"   we have {len(w['reviews'])} written reviews. The first one:")
    say(f"   \"{w['reviews'][0]['text']}\"")

    def tokenize(text):
        result = None   # TODO[5.1]: lowercase the text, then split on spaces:
                        #            text.lower().split()
        need_value(result, "5.1", "return text.lower().split()  (as one expression)")
        return result

    toks = tokenize("The Pad Thai was AMAZING")
    checkpoint("5.1", toks == ["the", "pad", "thai", "was", "amazing"],
               'tokenizer works: "...AMAZING" -> [..., "amazing"].',
               "Lowercase FIRST, then .split(). Both are needed.")

    # Our corpus is ALL the text the company owns: every review PLUS the
    # help pages (Assignment 9's chatbot will thank us for including them).
    faq_texts = [title.lower() + " " + txt.lower()
                 for _sid, title, txt in w["faq_docs"]]
    all_token_lists = ([tokenize(r["text"]) for r in w["reviews"]] +
                       [tokenize(t) for t in faq_texts])

    # Keep words that occur at least 3 times (rarer ones have too little
    # evidence to learn from) — and drop "glue words" (the, was, and...).
    # Glue words appear next to EVERYTHING, so they carry no meaning signal;
    # dropping them is standard practice and sharpens every vector we build.
    STOPWORDS = {"the", "a", "an", "was", "is", "are", "and", "or", "i",
                 "it", "to", "of", "for", "in", "on", "my", "with", "at",
                 "this", "that", "but", "not", "so", "very", "too", "will",
                 "you", "your", "can", "be", "we", "our", "from", "when",
                 "again", "ever", "had", "has", "have", "as", "by", "one",
                 "many", "more", "up", "out", "only", "if", "do", "how"}
    freq = {}
    for toks in all_token_lists:
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    vocab = sorted(t for t, c in freq.items()
                   if c >= 3 and t not in STOPWORDS)
    idx = {t: i for i, t in enumerate(vocab)}
    say(f"   vocabulary: {len(vocab)} meaning-carrying words appear 3+ times")

    # ----------------------------------------------------------------------
    # STEP 5.2 — Count neighbors: the co-occurrence table
    #   For each word, look at the words up to WINDOW positions away and
    #   count each sighting. WINDOW controls what "keeps company" means:
    #   1 = only touching words, 5 = the whole phrase around it.
    # ----------------------------------------------------------------------
    say("", "STEP 5.2 — count which words hang out together")

    WINDOW = None   # TODO[5.2]: use 2 (a good default: 2 words each side)

    need_value(WINDOW, "5.2", "WINDOW = 2")
    checkpoint("5.2", 1 <= WINDOW <= 5, f"window of ±{WINDOW} words.",
               "Any value 1..5 is legal; 2 is the sweet spot here.")

    C = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    for toks in all_token_lists:
        for i, t in enumerate(toks):
            if t not in idx:
                continue
            lo, hi = max(0, i - WINDOW), min(len(toks), i + WINDOW + 1)
            for j in range(lo, hi):
                if j != i and toks[j] in idx:
                    C[idx[t], idx[toks[j]]] += 1.0
    say(f"   co-occurrence table built: {C.shape[0]}x{C.shape[1]} counts")

    # ----------------------------------------------------------------------
    # STEP 5.3 — From counts to meaning scores to vectors  (read, don't edit)
    #   PPMI asks: "do these two words appear together MORE than luck would
    #   predict?" — common words like 'the' stop dominating. Then SVD (a
    #   matrix-math bottleneck, cousin of your A4 autoencoder) squeezes each
    #   word's 200+ scores down to 32 numbers.
    # ----------------------------------------------------------------------
    say("", "STEP 5.3 — squeeze counts into 32-number word vectors (given)")
    total = C.sum()
    p_word = C.sum(axis=1, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / total) / (p_word * p_word.T))
    ppmi = np.maximum(pmi, 0.0)           # keep only "MORE than chance"
    ppmi[~np.isfinite(ppmi)] = 0.0
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    DIM = 32
    word_vecs = (U[:, :DIM] * S[:DIM]).astype(np.float32)
    say(f"   done: every word is now {DIM} numbers.")

    # ----------------------------------------------------------------------
    # STEP 5.4 — Nearest neighbors: does the computer "get it"?
    #   Write a function that finds the k words most similar to a query.
    #   THE PLAN (translate each line to code):
    #     1. look up the query's vector:        word_vecs[idx[query]]
    #     2. cosine_sim it against EVERY row of word_vecs (a loop or
    #        list comprehension over range(len(vocab)))
    #     3. np.argsort the scores, reverse, take the top k+1
    #     4. drop the query word itself, return the rest as words
    # ----------------------------------------------------------------------
    say("", "STEP 5.4 — ask the vectors what goes with what")

    def nearest_words(query, k=6):
        # ===== TODO[5.4] BEGIN: implement the 4-step plan above (~4-6 lines)
        raise NotImplementedError("TODO 5.4")
        # ===== TODO[5.4] END =====

    try:
        test_nn = nearest_words("cold")
    except NotImplementedError:
        say("", "  ⏸  TODO[5.4]: follow the 4-step plan in the comment. Every",
            "     tool it needs (cosine_sim, np.argsort) you've already used.")
        raise NotDoneYet("5.4")

    for query in ("cold", "delicious", "refund"):
        say(f"   words near '{query}':  {', '.join(nearest_words(query))}")

    bad_words = {"soggy", "stale", "late", "freezing", "chewy", "inedible",
                 "disappointed", "arrived"}
    good_words = {"amazing", "fresh", "tasty", "wonderful", "hot", "great",
                  "loved", "flavor", "delicious", "perfectly"}
    nn_cold = set(nearest_words("cold", k=8))
    nn_yum  = set(nearest_words("delicious", k=8))
    checkpoint("5.4",
               len(nn_cold & bad_words) >= 1 and len(nn_yum & good_words) >= 1
               and "cold" not in nn_cold,
               "'cold' lives with the complaint words, 'delicious' with the "
               "praise words. It learned meaning from context alone.",
               "Neighbors look wrong. Check: reversed argsort? skipped the "
               "query word itself? similarity against word_vecs rows (not C)?")

    # ----------------------------------------------------------------------
    # STEP 5.5 — See the map (given) and think
    #   We project the 32 dimensions down to 2 and print a crude map.
    #   (In the real world you'd use t-SNE or UMAP for this picture.)
    # ----------------------------------------------------------------------
    say("", "STEP 5.5 — a 2-D map of word-meaning space")
    show_words = ["cold", "soggy", "stale", "late", "terrible", "refund",
                  "delicious", "amazing", "fresh", "tasty", "loved", "great",
                  "chicken", "beef", "tofu", "rice", "noodles", "curry",
                  "courier", "delivery", "arrived", "warm"]
    show_words = [s for s in show_words if s in idx]
    pts = word_vecs[[idx[s] for s in show_words]][:, :2]   # first 2 SVD dims
    gx = ((pts[:, 0] - pts[:, 0].min()) /
          (np.ptp(pts[:, 0]) + 1e-9) * 56).astype(int)
    gy = ((pts[:, 1] - pts[:, 1].min()) /
          (np.ptp(pts[:, 1]) + 1e-9) * 16).astype(int)
    grid = [[" "] * 68 for _ in range(18)]
    for word, x, y in zip(show_words, gx, gy):
        for c, ch in enumerate(word[:10]):
            if x + c < 68:
                grid[y][x + c] = ch
    say("   " + "-" * 68)
    for row in grid:
        say("   |" + "".join(row)[:66] + "|")
    say("   " + "-" * 68,
        "   (words that mean similar things should sit near each other)")

    think_check("5.5", THINK_5_5)

    ARTIFACTS[5] = {"vocab": vocab, "idx": idx, "word_vecs": word_vecs,
                    "tokenize": tokenize, "nearest_words": nearest_words}
    _mark_done(5)
    say("", "🎉 ASSIGNMENT 5 COMPLETE — marketing can auto-tag reviews now.",
        "   Next:  python ml2_final_project.py 6")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 6 — "READ REVIEWS LIKE A MODEL"
#   Course topic: Sequence models & attention (Week 7)
#   Difficulty: ▉▉▉▉▉▉░░░░        Time: ~1 week
#
# ==========================================================================
# THE STORY
#   Boss: "Support wants unhappy reviews routed to a human within minutes.
#   Build something that reads a review and says happy or unhappy — and
#   I want to SEE which words it based its decision on. No black boxes."
#
# YOUR GOAL
#   A sentiment classifier with ATTENTION: for every review it produces
#   a weight for each word — "how much did I look at this word?" — and
#   you'll print those weights and watch it stare at 'terrible'.
#
# THE BIG IDEA
#   Attention = a learned, differentiable version of "which parts of the
#   input matter right now?". It's THE mechanism inside transformers and
#   every modern LLM (Weeks 7 & 10). You're building the pocket version.
# ==========================================================================

THINK_6_5 = """
"""  # TODO[6.5]: compare the attention weights on a 5-star review and a
     # 1-star review. Which kinds of words get the weight, and why does
     # showing these weights make support trust the model more?


def assignment_6():
    banner("ASSIGNMENT 6 — READ REVIEWS LIKE A MODEL")
    w = world()
    a5 = need(5, assignment_5)
    tokenize = a5["tokenize"]

    # ----------------------------------------------------------------------
    # STEP 6.1 — Make the labels
    #   4-5 stars = happy (1.0), 1-2 stars = unhappy (0.0). We SKIP 3-star
    #   reviews: "meh" teaches the model nothing crisp.
    # ----------------------------------------------------------------------
    say("", "STEP 6.1 — label the reviews")

    HAPPY_MIN = None     # TODO[6.1a]: lowest rating that counts as happy -> 4
    UNHAPPY_MAX = None   # TODO[6.1b]: highest rating that counts as unhappy -> 2

    need_value(HAPPY_MIN, "6.1a", "HAPPY_MIN = 4")
    need_value(UNHAPPY_MAX, "6.1b", "UNHAPPY_MAX = 2")

    labeled = []
    for r in w["reviews"]:
        if r["rating"] >= HAPPY_MIN:
            labeled.append((r["text"], 1.0))
        elif r["rating"] <= UNHAPPY_MAX:
            labeled.append((r["text"], 0.0))
    n_happy = sum(1 for _, lab in labeled if lab == 1.0)
    say(f"   {len(labeled)} labeled reviews ({n_happy} happy, "
        f"{len(labeled) - n_happy} unhappy) — 3-star reviews skipped")
    checkpoint("6.1", HAPPY_MIN == 4 and UNHAPPY_MAX == 2 and len(labeled) > 700,
               "labels ready.",
               "Use 4 and 2. Three-star reviews must fall through both tests.")

    # ----------------------------------------------------------------------
    # STEP 6.2 — Same-length sequences
    #   Networks want rectangular data, but reviews have different lengths.
    #   Standard fix: cut long ones, PAD short ones with a special slot 0.
    # ----------------------------------------------------------------------
    say("", "STEP 6.2 — pad every review to the same length")

    MAXLEN = None   # TODO[6.2]: max words to keep per review — use 14

    need_value(MAXLEN, "6.2", "MAXLEN = 14")
    checkpoint("6.2", 8 <= MAXLEN <= 24, f"reviews clipped/padded to {MAXLEN} words.",
               "Anything 8..24 works; 14 fits most of our reviews.")

    # slot 0 = padding, so every real word gets its A5 index + 1
    vocab_idx = a5["idx"]

    def encode(text):
        ids = [vocab_idx[t] + 1 for t in tokenize(text) if t in vocab_idx]
        ids = ids[:MAXLEN]
        return ids + [0] * (MAXLEN - len(ids))

    X = torch.tensor([encode(t) for t, _ in labeled])
    y = torch.tensor([lab for _, lab in labeled])
    n_train = int(len(X) * 0.8)                     # A2 habits: hold data out!
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    # ----------------------------------------------------------------------
    # STEP 6.3 — Build attention (the heart of this assignment)
    #   For each word i the model computes a raw score s_i ("how relevant
    #   does this word look?"). Your job is the two-step finish:
    #     a) weights = softmax of the scores  -> positive, sum to 1
    #        (torch.softmax(scores, dim=1) — dim=1 means "across the words
    #        of each review", not across the batch!)
    #     b) summary = weighted sum of word vectors:
    #        (weights.unsqueeze(-1) * vectors).sum(dim=1)
    #   That summary — attention-pooled, not just averaged — goes to the
    #   final yes/no layer.
    # ----------------------------------------------------------------------
    say("", "STEP 6.3 — write the attention pooling")

    class AttentionClassifier(nn.Module):
        def __init__(self, vocab_size, dim=32):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.score = nn.Linear(dim, 1)      # word vector -> raw score
            self.out = nn.Linear(dim, 1)        # summary -> happy/unhappy

        def forward(self, ids):
            vectors = self.emb(ids)                       # (batch, words, dim)
            scores = self.score(vectors).squeeze(-1)      # (batch, words)
            scores = scores.masked_fill(ids == 0, -1e9)   # padding gets no say
            # ===== TODO[6.3] BEGIN: two lines, as described above:
            #   weights = ...softmax...
            #   summary = ...weighted sum...
            raise NotImplementedError("TODO 6.3")
            # ===== TODO[6.3] END =====
            return self.out(summary).squeeze(-1), weights

    torch.manual_seed(SEED)
    model = AttentionClassifier(len(a5["vocab"]) + 1)
    try:
        _logit, _wts = model(Xtr[:4])
    except NotImplementedError:
        say("", "  ⏸  TODO[6.3]: two lines. (a) softmax over dim=1;",
            "     (b) multiply weights (unsqueezed) by vectors, .sum(dim=1).")
        raise NotDoneYet("6.3")
    checkpoint("6.3",
               _wts.shape == (4, MAXLEN)
               and torch.allclose(_wts.sum(dim=1), torch.ones(4), atol=1e-4),
               "attention weights are positive and sum to 1 per review.",
               "weights must be shaped (batch, words) and each row must sum "
               "to 1 — softmax over dim=1, not dim=0.")

    # ----------------------------------------------------------------------
    # STEP 6.4 — Train and evaluate (loop is given — you've earned it)
    # ----------------------------------------------------------------------
    say("", "STEP 6.4 — train")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(40):
        logits, _ = model(Xtr)
        loss = loss_fn(logits, ytr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 10 == 0:
            say(f"   epoch {epoch:>3}   loss = {loss.item():.3f}")

    with torch.no_grad():
        logits, weights = model(Xte)
        accuracy = ((logits > 0) == (yte > 0.5)).float().mean().item()
    say(f"   held-out accuracy: {accuracy:.1%}")
    checkpoint("6.4", accuracy > 0.9,
               f"{accuracy:.1%} — support can trust the routing.",
               "Should exceed 90%. Verify 6.3: weighted SUM (not mean), and "
               "the mask line stayed untouched.")

    # Show the receipts: which words did it look at?
    say("", "   where the model looked (higher % = more attention):")
    with torch.no_grad():
        for row in (0, 1, 2):
            ids = Xte[row]
            _, wts = model(ids.unsqueeze(0))
            words = [(a5["vocab"][i - 1], float(wt))
                     for i, wt in zip(ids.tolist(), wts[0].tolist()) if i != 0]
            words.sort(key=lambda p: -p[1])
            verdict = "HAPPY" if logits[row] > 0 else "UNHAPPY"
            say(f"   [{verdict:>7}] " +
                "  ".join(f"{wd}({wt:.0%})" for wd, wt in words[:4]))

    think_check("6.5", THINK_6_5)

    ARTIFACTS[6] = {"sentiment_model": model, "encode": encode}
    _mark_done(6)
    say("", "🎉 ASSIGNMENT 6 COMPLETE — and you can SHOW people why it decides.",
        "   Next:  python ml2_final_project.py 7")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 7 — "IS THAT PHOTO REALLY A PIZZA?"
#   Course topic: Convolutional neural networks (Week 8)
#   Difficulty: ▉▉▉▉▉▉▉░░░        Time: ~1 week
#
# ==========================================================================
# THE STORY
#   Boss: "Restaurants upload the wrong photos constantly — noodles on the
#   pizza page. Customers notice. Build something that checks the photo."
#
# YOUR GOAL
#   A convolutional neural network that classifies FlavorCart's dish
#   photos (tiny 16x16 grayscale images) as pizza / sushi / noodles.
#
# THE BIG IDEA
#   A Linear layer would treat pixel (3,7) and pixel (3,8) as total
#   strangers. A CONVOLUTION slides a small window across the image, so
#   the network learns local shapes — edges, corners, dots — and reuses
#   them everywhere in the image. That's why CNNs see.
# ==========================================================================

THINK_7_5 = """
"""  # TODO[7.5]: your CNN has ~1,500 weights; a plain Linear layer doing
     # the same job would need ~50,000. Where do the savings come from?
     # (Hint: the same 3x3 filter gets REUSED at every image position.)


def _ascii_image(img):
    """Render a 16x16 grayscale image with text shades."""
    shades = " .:-=+*#%@"
    rows = []
    for r in range(16):
        rows.append("".join(shades[min(int(v * 9.99), 9)] for v in img[r]))
    return rows


def assignment_7():
    banner("ASSIGNMENT 7 — IS THAT PHOTO REALLY A PIZZA?")
    w = world()
    images, labels = w["images"], w["image_labels"]
    classes = w["image_classes"]

    # ----------------------------------------------------------------------
    # STEP 7.1 — Look at the data with your own eyes. Always. (given)
    # ----------------------------------------------------------------------
    say("", "STEP 7.1 — the photos (16x16 pixels of pure cuisine)")
    picks = [int(np.flatnonzero(labels == k)[0]) for k in (0, 1, 2)]
    trios = [_ascii_image(images[p]) for p in picks]
    say("   " + "".join(f"{classes[k]:^18}" for k in (0, 1, 2)))
    for r in range(16):
        say("   " + "  ".join(t[r] for t in trios))

    # ----------------------------------------------------------------------
    # STEP 7.2 — Design the convolution layer
    #   nn.Conv2d(in_channels, out_channels, kernel_size) needs three answers:
    #     in_channels : color channels coming IN. Our photos are grayscale.
    #     out_channels: how many different filters (shape detectors) to
    #                   learn. We want 8.
    #     kernel_size : the window size. The classic small window is 3.
    # ----------------------------------------------------------------------
    say("", "STEP 7.2 — your first convolution")

    IN_CHANNELS  = None   # TODO[7.2a]: grayscale images have how many channels? (1)
    OUT_CHANNELS = None   # TODO[7.2b]: we want 8 filters
    KERNEL       = None   # TODO[7.2c]: the classic window size, 3

    need_value(IN_CHANNELS,  "7.2a", "Grayscale = 1 channel (RGB photos would be 3).")
    need_value(OUT_CHANNELS, "7.2b", "OUT_CHANNELS = 8")
    need_value(KERNEL,       "7.2c", "KERNEL = 3")
    checkpoint("7.2", (IN_CHANNELS, OUT_CHANNELS, KERNEL) == (1, 8, 3),
               "Conv2d(1, 8, 3): 8 little 3x3 shape-detectors, ready to learn.",
               "Grayscale in (1), 8 filters out, 3x3 window.")

    class DishCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL)
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 2 * 2, 3)     # 3 classes out

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))   # conv -> shrink
            x = self.pool(torch.relu(self.conv2(x)))   # conv -> shrink again
            return self.fc(x.flatten(1))               # flatten -> classify

    # ----------------------------------------------------------------------
    # STEP 7.3 — Follow the shapes (the #1 practical CNN skill)
    #   Track one image through the network and predict the shape at each
    #   stop. Rules for our layers:
    #     conv 3x3 (no padding): size shrinks by 2      (16 -> 14)
    #     maxpool 2:             size halves, round down (14 -> 7)
    #   Channels: conv1 outputs 8, conv2 outputs 16.
    #   Format: (channels, height, width)
    # ----------------------------------------------------------------------
    say("", "STEP 7.3 — predict the tensor shapes")

    shape_after_conv1 = None   # TODO[7.3a]: 16x16 in -> (8, ?, ?)   e.g. (8, 14, 14)
    shape_after_pool1 = None   # TODO[7.3b]: then maxpool halves it -> (8, ?, ?)
    shape_after_conv2 = None   # TODO[7.3c]: conv 3x3 again, 16 filters -> (16, ?, ?)
    shape_after_pool2 = None   # TODO[7.3d]: final pool (round DOWN) -> (16, ?, ?)

    need_value(shape_after_conv1, "7.3a", "Conv 3x3 shrinks 16 to 14: (8, 14, 14)")
    need_value(shape_after_pool1, "7.3b", "Pool halves 14 to 7.")
    need_value(shape_after_conv2, "7.3c", "Conv shrinks 7 to 5, with 16 channels.")
    need_value(shape_after_pool2, "7.3d", "5 halves and rounds DOWN to 2.")

    torch.manual_seed(SEED)
    cnn = DishCNN()
    x = torch.tensor(images[:1]).unsqueeze(1)          # (1 image, 1 channel, 16, 16)
    h1 = torch.relu(cnn.conv1(x)); p1 = cnn.pool(h1)
    h2 = torch.relu(cnn.conv2(p1)); p2 = cnn.pool(h2)
    actual = [tuple(t.shape[1:]) for t in (h1, p1, h2, p2)]
    say("   your predictions vs reality:")
    names = ["after conv1", "after pool1", "after conv2", "after pool2"]
    guesses = [shape_after_conv1, shape_after_pool1, shape_after_conv2, shape_after_pool2]
    for name, guess, real in zip(names, guesses, actual):
        say(f"      {name:<12} you said {str(guess):<13} actual {real}")
    checkpoint("7.3", all(tuple(g) == r for g, r in zip(guesses, actual)),
               "you can now trace shapes through any CNN. Superpower unlocked.",
               "conv 3x3 (no padding): -2 per side stays -2 total (16->14, "
               "7->5). maxpool 2: halve and round down (5->2).")

    # ----------------------------------------------------------------------
    # STEP 7.4 — Train it (given) and grade it on held-out photos
    # ----------------------------------------------------------------------
    say("", "STEP 7.4 — train the CNN")
    X = torch.tensor(images).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    Xtr, ytr, Xte, yte = X[:480], y[:480], X[480:], y[480:]

    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()          # the loss for pick-one-of-N
    for epoch in range(30):
        loss = loss_fn(cnn(Xtr), ytr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 10 == 0:
            say(f"   epoch {epoch:>3}   loss = {loss.item():.3f}")

    with torch.no_grad():
        pred = cnn(Xte).argmax(dim=1)
        accuracy = (pred == yte).float().mean().item()
    say(f"   held-out accuracy: {accuracy:.1%} on {len(yte)} unseen photos")
    checkpoint("7.4", accuracy > 0.9,
               f"{accuracy:.1%} — wrong-photo uploads get caught at the door.",
               "Should be over 90%. Make sure 7.2's numbers are (1, 8, 3).")

    # Wrong-photo demo: the actual product feature
    with torch.no_grad():
        probs = torch.softmax(cnn(Xte[:1]), dim=1)[0]
    say("   example check — restaurant uploads photo, claims it's 'sushi':",
        "      model says: " + ", ".join(f"{classes[k]} {probs[k]:.0%}"
                                         for k in range(3)),
        f"      true class: {classes[int(yte[0])]}")

    think_check("7.5", THINK_7_5)

    ARTIFACTS[7] = {"cnn": cnn}
    _mark_done(7)
    say("", "🎉 ASSIGNMENT 7 COMPLETE — no more noodles on the pizza page.",
        "   Next:  python ml2_final_project.py 8")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 8 — "AUTOCOMPLETE THE REVIEW"
#   Course topic: Generative models & how LLMs work (Weeks 9-10)
#   Difficulty: ▉▉▉▉▉▉▉▉░░        Time: ~1-1.5 weeks
#
# ==========================================================================
# THE STORY
#   Boss: "Typing reviews on a phone is painful — half our customers give
#   up mid-sentence. Could the app suggest how the sentence continues?"
#
# YOUR GOAL
#   Train a tiny language model that generates FlavorCart-style review
#   text one character at a time — then steer its personality with a
#   single number: TEMPERATURE.
#
# THE BIG IDEA
#   This is EXACTLY how ChatGPT-class models work, shrunk a million-fold:
#   read the context, predict a probability for every possible next token,
#   sample one, repeat. Everything you observe here — including the
#   nonsense it sometimes writes — has a big-sibling version in real LLMs.
# ==========================================================================

THINK_8_4 = """
"""  # TODO[8.4]: describe what changed between temperature 0.3, 0.9 and
     # 1.6. If you were shipping the autocomplete feature, which would you
     # pick, and what does that say about the creativity/reliability trade?


def assignment_8():
    banner("ASSIGNMENT 8 — AUTOCOMPLETE THE REVIEW")
    w = world()

    text = "\n".join(r["text"] for r in w["reviews"])
    say(f"   training text: every FlavorCart review, {len(text):,} characters total")

    # ----------------------------------------------------------------------
    # STEP 8.1 — The vocabulary of a character-level model
    #   Our "tokens" today are single CHARACTERS. The vocabulary is simply
    #   every distinct character in the text, sorted (so it's stable).
    #   Toolbox:  set(text) -> distinct chars,  sorted(...) -> stable list
    # ----------------------------------------------------------------------
    say("", "STEP 8.1 — build the character vocabulary")

    chars = None   # TODO[8.1]: sorted(set(text))

    need_value(chars, "8.1", "chars = sorted(set(text))")
    checkpoint("8.1", isinstance(chars, list) and len(chars) < 50
               and " " in chars and chars == sorted(chars),
               f"{len(chars)} distinct characters — that's the whole alphabet "
               "this model will ever know.",
               "sorted(set(text)) — set() finds distinct chars, sorted() "
               "returns them as a stable list.")

    c2i = {c: i for i, c in enumerate(chars)}
    encoded = np.array([c2i[c] for c in text], dtype=np.int64)

    # ----------------------------------------------------------------------
    # STEP 8.2 — Training pairs: (24 characters in, the 25th out)
    #   A language model's whole education is this one exercise, repeated
    #   millions of times: HERE is some context — WHAT comes next?
    #   Slicing reminder: encoded[i : i + L] grabs L items starting at i.
    # ----------------------------------------------------------------------
    say("", "STEP 8.2 — build the guess-the-next-character exercises")
    L = 24                                   # context window (characters)
    starts = np.arange(0, len(encoded) - L - 1, 3)   # every 3rd position

    X_list, y_list = [], []
    for i in starts:
        context = None    # TODO[8.2a]: the L characters starting at i ->  encoded[i : i + L]
        target  = None    # TODO[8.2b]: the single character right AFTER them -> encoded[i + L]
        need_value(context, "8.2a", "encoded[i : i + L]")
        need_value(target,  "8.2b", "encoded[i + L]")
        X_list.append(context); y_list.append(target)

    X = torch.tensor(np.array(X_list))
    y = torch.tensor(np.array(y_list))
    checkpoint("8.2", X.shape[1] == L and len(X) == len(y)
               and int(X[0][-1]) == int(encoded[L - 1]) and int(y[0]) == int(encoded[L]),
               f"{len(X):,} exercises ready: see {L} chars, guess char {L + 1}.",
               "context = encoded[i : i + L] and target = encoded[i + L]. "
               "Off-by-one errors are a rite of passage — check both ends.")

    # ----------------------------------------------------------------------
    # STEP 8.3 — The model (given) and YOUR sampling function
    #   The network (embedding -> LSTM -> linear) outputs one score per
    #   character in the vocabulary: the LOGITS. Turning logits into the
    #   next character is where temperature lives:
    #       probs = softmax(logits / T)
    #   T < 1  sharpens: the favorite gets even more likely (plays it safe)
    #   T > 1  flattens: underdogs get real chances (gets creative/weird)
    #   Then torch.multinomial(probs, 1) rolls the weighted dice.
    # ----------------------------------------------------------------------
    say("", "STEP 8.3 — train, then write the sampler")

    class TinyLM(nn.Module):
        def __init__(self, vocab_size, dim=48):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, dim)
            self.lstm = nn.LSTM(dim, dim, batch_first=True)
            self.out = nn.Linear(dim, vocab_size)

        def forward(self, ids):
            h, _ = self.lstm(self.emb(ids))
            return self.out(h[:, -1, :])       # score every possible next char

    torch.manual_seed(SEED)
    lm = TinyLM(len(chars))
    optimizer = torch.optim.Adam(lm.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    say("   training (about a minute of real deep learning — watch the loss)...")
    n_batches = 0
    for epoch in range(3):
        perm = torch.randperm(len(X))
        for b in range(0, len(X), 512):
            batch = perm[b:b + 512]
            loss = loss_fn(lm(X[batch]), y[batch])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            n_batches += 1
            if n_batches % 30 == 0:
                say(f"   batch {n_batches:>4}   loss = {loss.item():.3f}")
    final_loss = loss.item()
    checkpoint("8.3a", final_loss < 1.5,
               f"loss {final_loss:.2f} — the model has learned real English-ish habits.",
               "Loss should get below 1.5. Did steps 8.1/8.2 pass legitimately?")

    def sample_next_char(context_ids, T):
        """Given context (tensor of ids), pick the next char id at temp T."""
        with torch.no_grad():
            logits = lm(context_ids.unsqueeze(0))[0]

        probs = None   # TODO[8.3]: torch.softmax(logits / T, dim=-1)

        need_value(probs, "8.3", "probs = torch.softmax(logits / T, dim=-1)")
        return int(torch.multinomial(probs, 1))

    def generate(prompt, n=90, T=0.8):
        torch.manual_seed(SEED)                     # reproducible "creativity"
        ids = [c2i.get(c, 0) for c in prompt]
        for _ in range(n):
            ctx = torch.tensor(ids[-L:])
            ids.append(sample_next_char(ctx, T))
        return "".join(chars[i] for i in ids)

    # quick sanity: student's line must give a real probability distribution
    _p = sample_next_char(X[0], 1.0)      # raises NotDoneYet if TODO unfilled
    checkpoint("8.3b", 0 <= _p < len(chars),
               "sampler works — it returns a valid character id.",
               "probs must come from softmax(logits / T) along dim=-1.")

    # ----------------------------------------------------------------------
    # STEP 8.4 — Meet your model's three personalities
    # ----------------------------------------------------------------------
    say("", "STEP 8.4 — the same model at three temperatures")
    for T in (0.3, 0.9, 1.6):
        out = generate("the pad thai was ", n=80, T=T)
        say(f"   T={T}:  \"{out}\"", "")
    say("   (low T repeats its safest phrases; high T takes risks — and",
        "    sometimes writes gibberish. Real LLM APIs expose this exact knob.)")

    think_check("8.4", THINK_8_4)

    # ----------------------------------------------------------------------
    # STEP 8.5 — Why real LLMs don't use characters (a 60-second detour)
    #   Compare how many steps it takes to write the same review:
    # ----------------------------------------------------------------------
    say("", "STEP 8.5 — characters vs words vs subwords")
    sample_review = w["reviews"][0]["text"]

    n_char_tokens = None   # TODO[8.5a]: characters in sample_review -> len(sample_review)
    n_word_tokens = None   # TODO[8.5b]: words in it -> len(sample_review.split())

    need_value(n_char_tokens, "8.5a", "len(sample_review)")
    need_value(n_word_tokens, "8.5b", "len(sample_review.split())")
    checkpoint("8.5", n_char_tokens == len(sample_review)
               and n_word_tokens == len(sample_review.split()),
               f"same review: {n_char_tokens} character-steps vs "
               f"{n_word_tokens} word-steps. Real LLMs split the difference "
               "with SUBWORD tokens (~1.3x the word count).",
               "Two len() calls — one on the string, one on .split().")

    ARTIFACTS[8] = {"generate": generate, "lm": lm}
    _mark_done(8)
    say("", "🎉 ASSIGNMENT 8 COMPLETE — you have trained a (very) small language model.",
        "   Next:  python ml2_final_project.py 9")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 9 — "THE SUPPORT CHATBOT"
#   Course topic: Retrieval-Augmented Generation / RAG (Weeks 11-12)
#   Difficulty: ▉▉▉▉▉▉▉▉▉░        Time: ~1.5 weeks
#
# ==========================================================================
# THE STORY
#   Boss: "Support answers the same 12 questions all day. We have help
#   pages for all of them! Build a bot that finds the right help page and
#   answers from it. And it must NOT make things up — that's a lawsuit."
#
# YOUR GOAL
#   A real RAG pipeline, the architecture behind most serious chatbots:
#     RETRIEVE the most relevant document (using YOUR embeddings from A5
#     and YOUR cosine similarity from A3), then GENERATE an answer that
#     only uses what the document says.
#
#   We use a tiny stand-in "LLM" so everything runs offline — but the
#   pipeline you build is, piece for piece, the production architecture.
#   (Optional extension at the end: swap in a real LLM API.)
# ==========================================================================

THINK_9_5 = """
"""  # TODO[9.5]: when you asked about the moon, the first bot confidently
     # answered from an irrelevant page. Real LLMs do the same thing (it's
     # called hallucination). Why does "retrieve first, and refuse when
     # retrieval is weak" reduce making-stuff-up? What does it cost us?


def assignment_9():
    banner("ASSIGNMENT 9 — THE SUPPORT CHATBOT")
    w = world()
    a3 = need(3, assignment_3)
    a5 = need(5, assignment_5)
    cosine_sim, tokenize = a3["cosine_sim"], a5["tokenize"]
    idx, word_vecs = a5["idx"], a5["word_vecs"]

    docs = [{"doc_id": sid, "title": title, "text": txt}
            for sid, title, txt in w["faq_docs"]]
    say(f"   knowledge base: {len(docs)} help pages "
        f"({', '.join(d['doc_id'] for d in docs[:5])}, ...)")

    # First, a small helper we build FOR you: a "specialness" score per word.
    # A word that appears in only ONE help page (like 'vegan' or 'cancel')
    # is a strong clue about which page a question belongs to. A word that
    # appears in most pages (like 'order') tells us almost nothing. This
    # idea is called IDF (inverse document frequency) and it powers real
    # search engines — including the retrieval half of production RAG.
    n_docs_with = {}
    for d in docs:
        for t in set(tokenize(d["title"].lower() + " " + d["text"].lower())):
            n_docs_with[t] = n_docs_with.get(t, 0) + 1
    idf = {t: float(np.log(len(docs) / n)) for t, n in n_docs_with.items()}
    say("   word specialness (idf): " +
        ", ".join(f"{t}={idf.get(t, 1.0):.1f}"
                  for t in ("vegan", "cancel", "refund", "order", "page")))

    # Each document, reduced to its set of distinct words (computed for you):
    doc_words = [set(tokenize(d["title"].lower() + " " + d["text"].lower()))
                 for d in docs]

    # ----------------------------------------------------------------------
    # STEP 9.1 — Score one document against one question
    #   The classic search-engine recipe (TF-IDF), pocket-sized: a document
    #   scores high if it contains the question's SPECIAL words.
    #   THE PLAN:
    #     1. q_words = set(tokenize(question))     <- DISTINCT words only
    #     2. add up idf[t] for every t in q_words that is in doc_words[i]
    #        (sum(... for t in q_words if t in doc_words[i]) does this)
    #     3. return that sum
    #   Shared special word ('vegan') -> big boost. Shared generic word
    #   ('order') -> almost nothing. No shared words -> score 0.
    # ----------------------------------------------------------------------
    say("", "STEP 9.1 — score a help page against a question")

    def keyword_score(question, i):
        # ===== TODO[9.1] BEGIN: implement the 3-step plan (~2 lines)
        raise NotImplementedError("TODO 9.1")
        # ===== TODO[9.1] END =====

    try:
        s_refund = keyword_score("how do i get a refund for my cold order", 0)
        s_tip = keyword_score("how do i get a refund for my cold order", 7)
    except NotImplementedError:
        say("", "  ⏸  TODO[9.1]: follow the 3-step plan. It's a set + a sum —",
            "     no numpy needed for this one.")
        raise NotDoneYet("9.1")
    say(f"   refund question vs 'Refund Policy' page: {s_refund:.2f}",
        f"   refund question vs 'Tipping' page:       {s_tip:.2f}")
    checkpoint("9.1", s_refund > s_tip and s_refund > 2.0,
               "special shared words pull the right page ahead. That's search.",
               "Use DISTINCT question words (a set), add idf[t] only for "
               "words the document actually contains.")

    # ----------------------------------------------------------------------
    # STEP 9.2 — RETRIEVE: rank every help page for this question
    #   Score all 12 docs, best first, top k — the argsort pattern you have
    #   now used three times (A3 dishes, A5 words). This time, NO hints.
    #   Spec: return a list of (doc, score) pairs, highest score first.
    # ----------------------------------------------------------------------
    say("", "STEP 9.2 — the retriever (your third argsort rodeo — no hints)")

    def retrieve(question, k=2):
        # ===== TODO[9.2] BEGIN: ~4 lines. Return list of (doc, score) pairs.
        raise NotImplementedError("TODO 9.2")
        # ===== TODO[9.2] END =====

    try:
        top = retrieve("how do i get a refund for my cold order")
    except NotImplementedError:
        say("", "  ⏸  TODO[9.2]: keyword_score the question against every doc",
            "     index, np.argsort the scores, reverse, take k, return",
            "     [(docs[i], scores[i]) for those i].")
        raise NotDoneYet("9.2")
    say("   retrieve('...refund...cold order') ->",
        *[f"      {score:.2f}  {doc['title']}" for doc, score in top])
    checkpoint("9.2", len(top) == 2 and top[0][1] >= top[1][1]
               and isinstance(top[0][0], dict)
               and top[0][0]["doc_id"] == "refunds",
               "retriever returns ranked (document, score) pairs.",
               "Return exactly k pairs, best first: sort scores DESCENDING.")

    # A glimpse of the DENSE upgrade (read-only): production systems also
    # embed docs and questions as vectors (like your A5 word vectors, but
    # from billion-word models) and retrieve by cosine similarity, catching
    # synonyms that keyword search misses ("money back" -> refunds). With
    # only ~1300 reviews our embeddings are too fuzzy for that job — watch:
    q_vec = np.mean([word_vecs[idx[t]] for t in
                     tokenize("how do i get a refund for my cold order")
                     if t in idx], axis=0)
    d_vec = np.mean([word_vecs[idx[t]] for t in doc_words[0] if t in idx], axis=0)
    say("", f"   (dense preview: embedding similarity question<->refund page = "
        f"{cosine_sim(q_vec, d_vec):.2f} — real systems get sharp numbers by "
        "training on billions of words, not 1,300 reviews)")

    # ----------------------------------------------------------------------
    # STEP 9.3 — GENERATE: assemble the prompt, let the 'LLM' answer
    #   A real RAG system pastes the retrieved text into the LLM's prompt.
    #   The model can only answer from what's in front of it — that's the
    #   whole safety story of RAG. Our toy LLM does what a good aligned
    #   LLM does with such a prompt: answer strictly from the context.
    # ----------------------------------------------------------------------
    say("", "STEP 9.3 — wire retrieval into the generator")

    def toy_llm(prompt):
        """A stand-in LLM: answers using ONLY the context in the prompt."""
        context = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
        question = prompt.split("QUESTION:")[1].strip()
        first_doc = context.split("\n")[0]
        title, body = first_doc.split(" || ")
        return f"According to our '{title}' page: {body}"

    def answer(question):
        results = retrieve(question, k=2)
        context = "\n".join(f"{d['title']} || {d['text']}" for d, _ in results)

        prompt = None   # TODO[9.3]: build the prompt string, exactly:
                        #   "CONTEXT:\n" + context + "\nQUESTION: " + question

        need_value(prompt, "9.3", '"CONTEXT:\\n" + context + "\\nQUESTION: " + question')
        return toy_llm(prompt), results

    reply, _ = answer("how do i get a refund for my cold order")
    say(f"   BOT: {reply[:110]}...")
    checkpoint("9.3", reply.startswith("According to our") and "refund" in reply.lower(),
               "the bot answers from the retrieved page — not from vibes.",
               "The prompt must contain CONTEXT: and QUESTION: markers "
               "exactly as specified (the toy LLM parses them).")

    # ----------------------------------------------------------------------
    # STEP 9.4 — Grade the bot on support's five most common questions
    # ----------------------------------------------------------------------
    say("", "STEP 9.4 — the support quiz (given)")
    quiz = [
        ("how do i get a refund for my cold order",        "refunds"),
        ("which dishes are vegan or plant based",          "vegan"),
        ("how spicy are the dishes can i order mild",      "spice"),
        ("i need to cancel my order before cooking starts", "cancel"),
        ("where is my courier and how do i track my order", "tracking"),
    ]
    hits = 0
    for question, want in quiz:
        results = retrieve(question, k=1)
        got = results[0][0]["doc_id"]
        mark = "✅" if got == want else "❌"
        hits += (got == want)
        say(f"   {mark} \"{question}\"  ->  {got}")
    checkpoint("9.4", hits >= 4,
               f"{hits}/5 questions routed to the right help page.",
               "At least 4/5 should hit. Check 9.1 includes the title in the "
               "doc text (we pass it in), and 9.2 sorts best-first.")

    # ----------------------------------------------------------------------
    # STEP 9.5 — Break it, then make it honest
    #   Ask something the help pages can't answer, and watch the bot
    #   confidently answer anyway. Then add the guardrail.
    # ----------------------------------------------------------------------
    say("", "STEP 9.5 — the hallucination demo, and the fix")
    moon_q = "do you deliver food to the moon"
    reply, results = answer(moon_q)
    say(f"   Q: {moon_q}",
        f"   BOT (no guardrail): {reply[:100]}...",
        f"   ...its best retrieval score was only {results[0][1]:.2f}. It answered anyway!")

    MIN_SIM = None   # TODO[9.5g]: refuse when the best retrieval score is
                     #   below this. Look at the scores: the five real
                     #   questions all scored above 3, the moon question
                     #   under 2. MIN_SIM = 2.5 splits them cleanly.

    need_value(MIN_SIM, "9.5g", "MIN_SIM = 2.5")

    def safe_answer(question):
        results = retrieve(question, k=2)
        # written as "not (>= MIN_SIM)" so an undefined similarity (a
        # question with NO recognizable words) also refuses — fail safe.
        if not (results[0][1] >= MIN_SIM):
            return ("I could not find this in our help pages, so I will not "
                    "guess. Contacting a human agent for you."), results
        return answer(question)[0], results

    reply2, _ = safe_answer(moon_q)
    reply3, _ = safe_answer("how do i get a refund for my cold order")
    say(f"   BOT (guardrail):    {reply2[:80]}...")
    checkpoint("9.5", reply2.startswith("I could not find")
               and reply3.startswith("According to our"),
               "the bot now refuses what it can't source. Legal is thrilled.",
               "With MIN_SIM = 2.5 the moon question must refuse and the "
               "refund question must still answer.")

    think_check("9.5", THINK_9_5)

    # OPTIONAL EXTENSION (no checkpoint): replace toy_llm() with a real API.
    # The pipeline stays identical — only the generator swaps. E.g.:
    #   client.messages.create(model="claude-sonnet-5", max_tokens=300,
    #       messages=[{"role": "user", "content": prompt}])
    # Try it if you have an API key. Notice YOU still control the context.

    ARTIFACTS[9] = {"retrieve": retrieve, "safe_answer": safe_answer,
                    "keyword_score": keyword_score}
    _mark_done(9)
    say("", "🎉 ASSIGNMENT 9 COMPLETE — support tickets drop 40% (boss's estimate).",
        "   Next (the finale):  python ml2_final_project.py 10")


# ==========================================================================
# ==========================================================================
#
#   ASSIGNMENT 10 — "SHIP IT"  (CAPSTONE)
#   Course topic: Evaluation, LLM-as-judge, agents (Weeks 13-15)
#   Difficulty: ▉▉▉▉▉▉▉▉▉▉        Time: ~1.5 weeks
#
# ==========================================================================
# THE STORY
#   Boss: "Board meeting Friday. I need three things: NUMBERS proving the
#   ML systems work, ONE assistant that can use all of them, and your
#   write-up. Ship it."
#
# YOUR GOAL
#   1. An evaluation scoreboard for every model you built (Weeks 13's
#      lesson: a model without an honest metric is a rumor).
#   2. A grader that judges the chatbot's answers automatically — the
#      LLM-as-judge idea, pocket-sized.
#   3. An AGENT: one front door that reads a request, DECIDES which of
#      your tools fits, calls it, and reports what it did (Week 14).
#   4. Your final report (Week 15: where does this go next?).
#
#   Note this assignment hands you almost no code. You have built every
#   piece before. This is the exam that feels like a job.
# ==========================================================================

REPORT = {
    "biggest_win":     "",   # TODO[10.5a]: which assignment result impressed you most, and why?
    "hardest_bug":     "",   # TODO[10.5b]: the bug/step that fought you hardest — and how you beat it
    "favorite_model":  "",   # TODO[10.5c]: of all the models you trained, which one and why?
    "one_improvement": "",   # TODO[10.5d]: if you had two more weeks, what would you build/fix?
    "advice":          "",   # TODO[10.5e]: one paragraph of advice for the next student
}


def assignment_10():
    banner("ASSIGNMENT 10 — SHIP IT (CAPSTONE)")
    w = world()

    # ----------------------------------------------------------------------
    # STEP 10.1 — The scoreboard
    #   Write the two universal metric functions yourself. Specs:
    #     accuracy(preds, targets):  fraction where they match.
    #        np.mean of an == comparison does this in one line.
    #     mean_abs_error(preds, targets): average of absolute differences.
    #        np.abs and np.mean do this in one line.
    # ----------------------------------------------------------------------
    say("", "STEP 10.1 — write your metrics, then score EVERYTHING")

    def accuracy(preds, targets):
        # ===== TODO[10.1a] BEGIN: one line
        raise NotImplementedError("TODO 10.1a")
        # ===== TODO[10.1a] END =====

    def mean_abs_error(preds, targets):
        # ===== TODO[10.1b] BEGIN: one line
        raise NotImplementedError("TODO 10.1b")
        # ===== TODO[10.1b] END =====

    try:
        ok = (abs(accuracy(np.array([1, 0, 1]), np.array([1, 1, 1])) - 2 / 3) < 1e-9
              and abs(mean_abs_error(np.array([3.0, 5.0]), np.array([4.0, 3.0])) - 1.5) < 1e-9)
    except NotImplementedError:
        say("", "  ⏸  TODO[10.1a/b]: two one-liners. accuracy: np.mean(preds ==",
            "     targets). MAE: np.mean(np.abs(preds - targets)).")
        raise NotDoneYet("10.1")
    checkpoint("10.1a", ok, "your metrics agree with the reference cases.",
               "accuracy([1,0,1],[1,1,1]) must be 0.667; MAE([3,5],[4,3]) "
               "must be 1.5.")

    say("   re-running your systems to collect honest numbers "
        "(this trains several models — give it a minute)...")
    a2 = need(2, assignment_2)
    a6 = need(6, assignment_6)
    a7 = need(7, assignment_7)
    a9 = need(9, assignment_9)

    # Rating model on ITS OWN test split (recomputed exactly as in A2):
    feats = need(1, assignment_1)["order_features"]
    X_all = np.array([feats(od, w) for od in w["orders"]], dtype=np.float32)
    y_all = np.array([od["rating"] for od in w["orders"]], dtype=np.float32)
    Xte = (X_all[3400:] - a2["mu"]) / a2["std"]
    with torch.no_grad():
        rating_preds = a2["rating_model"](torch.tensor(Xte)).squeeze().numpy()
    rating_mae = mean_abs_error(rating_preds, y_all[3400:])

    # Sentiment model, on fresh encodings of the last 20% of labeled reviews:
    labeled = [(r["text"], 1.0 if r["rating"] >= 4 else 0.0)
               for r in w["reviews"] if r["rating"] != 3]
    n_train = int(len(labeled) * 0.8)
    Xs = torch.tensor([a6["encode"](t) for t, _ in labeled[n_train:]])
    ys = np.array([lab for _, lab in labeled[n_train:]])
    with torch.no_grad():
        logits, _ = a6["sentiment_model"](Xs)
    sent_acc = accuracy((logits.numpy() > 0).astype(float), ys)

    # CNN on its held-out photos:
    Xi = torch.tensor(w["images"][480:]).unsqueeze(1)
    with torch.no_grad():
        img_preds = a7["cnn"](Xi).argmax(dim=1).numpy()
    img_acc = accuracy(img_preds, w["image_labels"][480:])

    say("",
        "   ┌──────────────────────────────────────────────────────────┐",
        "   │              FLAVORCART ML — BOARD SCOREBOARD             │",
        "   ├──────────────────────────────────────────────────────────┤",
        f"   │  Rating predictor (A2)     off by {rating_mae:4.2f} stars on average  │",
        f"   │  Review sentiment (A6)     {sent_acc:6.1%} accuracy               │",
        f"   │  Photo checker    (A7)     {img_acc:6.1%} accuracy               │",
        "   └──────────────────────────────────────────────────────────┘")
    checkpoint("10.1b", rating_mae < 0.9 and sent_acc > 0.85 and img_acc > 0.85,
               "every system clears its bar, measured on held-out data.",
               "If a number misses its bar, revisit that assignment — the "
               "scoreboard only reports what your models actually do.")

    # ----------------------------------------------------------------------
    # STEP 10.2 — Error analysis: look at your worst mistakes
    #   Metrics say HOW GOOD; error analysis says WHERE IT FAILS.
    #   Find the 3 test orders where the rating model missed by the most.
    #   You've used the argsort-reverse-slice pattern three times now.
    # ----------------------------------------------------------------------
    say("", "STEP 10.2 — the three worst misses")
    abs_errors = np.abs(rating_preds - y_all[3400:])

    worst3 = None   # TODO[10.2]: positions of the 3 LARGEST abs_errors

    need_value(worst3, "10.2", "The same pattern as A3's best3 — but on abs_errors.")
    checkpoint("10.2", list(np.sort(np.asarray(worst3))) ==
               list(np.sort(np.argsort(abs_errors)[::-1][:3])),
               "found them.",
               "np.argsort(abs_errors)[::-1][:3] — biggest errors first.")
    for i in worst3:
        od = w["orders"][3400 + int(i)]
        dish = w["dishes"][od["dish_id"]]
        say(f"      predicted {rating_preds[int(i)]:.1f}, actual {od['rating']} "
            f"— {dish['name']} ({dish['cuisine']}), delivery {od['delivery_min']:.0f} min")
    say("   (notice a pattern? THAT observation is what real ML teams "
        "put in their next sprint.)")

    # ----------------------------------------------------------------------
    # STEP 10.3 — Judge the chatbot automatically (LLM-as-judge, pocket size)
    #   Week 13's problem: how do you grade free-form text answers at
    #   scale? Modern answer: another model grades them. Ours is a rubric:
    #   an answer is GROUNDED if it actually uses the retrieved page.
    #   Spec: return True when at least 3 distinct words from the source
    #   document's text (words longer than 4 letters) appear in the answer.
    #   PLAN: source_words = {w for w in doc["text"].lower().split() if len(w) > 4}
    #         count how many of those appear in answer.lower()
    #         return count >= 3
    # ----------------------------------------------------------------------
    say("", "STEP 10.3 — the answer judge")

    def judge_grounded(answer_text, doc):
        # ===== TODO[10.3] BEGIN: implement the 3-line plan above
        raise NotImplementedError("TODO 10.3")
        # ===== TODO[10.3] END =====

    quiz = ["how do i get a refund for my cold order",
            "which dishes are vegan or plant based",
            "i need to cancel my order before cooking starts"]
    try:
        results = []
        for q in quiz:
            reply, retrieved = a9["safe_answer"](q)
            results.append(judge_grounded(reply, retrieved[0][0]))
    except NotImplementedError:
        say("", "  ⏸  TODO[10.3]: follow the 3-line plan in the comment.")
        raise NotDoneYet("10.3")
    fabricated = "Great news, we deliver everywhere in the solar system for free!"
    say("   judge says: " + ", ".join(f"answer{i+1}={'GROUNDED' if r else 'NOT GROUNDED'}"
                                      for i, r in enumerate(results)))
    checkpoint("10.3", all(results)
               and not judge_grounded(fabricated, {"text": FAQ_DOCS[0][2]}),
               "real answers pass, a fabricated answer fails. The judge works.",
               "All three real answers must be GROUNDED and the fake "
               "'solar system' answer must not. Count DISTINCT source words "
               "(a set) of length > 4.")

    # ----------------------------------------------------------------------
    # STEP 10.4 — The agent: one front door for the whole company
    #   An agent = a loop that READS a request, CHOOSES a tool, ACTS, and
    #   REPORTS. Your tools are the systems you built. Write the router.
    #   Spec — return exactly one of "recommend", "faq", "status":
    #     mentions of suggesting/recommending/what to order  -> "recommend"
    #     questions about policies (refund/vegan/cancel/...) -> "faq"
    #     "where is my order/courier", tracking a live order -> "status"
    #   Keyword if/elif is honest engineering here. Real agents do this
    #   with an LLM call — the LOOP around it is identical.
    # ----------------------------------------------------------------------
    say("", "STEP 10.4 — build the agent's router")

    def route(request):
        # ===== TODO[10.4] BEGIN: if/elif on keywords, return one of the
        #   three tool names. The test requests are printed below — make
        #   your rules handle all six.
        raise NotImplementedError("TODO 10.4")
        # ===== TODO[10.4] END =====

    def tool_recommend(request, customer_id=7):
        recs = need(3, assignment_3)["recommend"](customer_id)
        return "you might love: " + ", ".join(d["name"] for d in recs)

    def tool_faq(request):
        return a9["safe_answer"](request)[0][:90] + "..."

    def tool_status(request):
        return "your courier picked up the order 6 minutes ago — ETA 18 minutes"

    TOOLS = {"recommend": tool_recommend, "faq": tool_faq, "status": tool_status}
    requests = [
        ("can you suggest something new for dinner tonight", "recommend"),
        ("what should i order if i love spicy food",         "recommend"),
        ("how do i get a refund my food arrived cold",       "faq"),
        ("are there vegan dishes on the menu",               "faq"),
        ("where is my order right now",                      "status"),
        ("track my courier please",                          "status"),
    ]
    try:
        routed = [(req, route(req)) for req, _ in requests]
    except NotImplementedError:
        say("", "  ⏸  TODO[10.4]: write the if/elif router. Look at the six",
            "     test requests in the code just below the TODO block.")
        raise NotDoneYet("10.4")

    say("   the agent at work:")
    correct = 0
    for (req, want), (_, got) in zip(requests, routed):
        ok = got == want
        correct += ok
        say(f"   {'✅' if ok else '❌'} \"{req}\"",
            f"        -> tool: {got}")
        if got in TOOLS:
            say(f"        -> {TOOLS[got](req)}")
    checkpoint("10.4", correct == len(requests),
               "6/6 requests reached the right tool. That's an agent.",
               "Route by keywords: suggest/recommend/should-i-order -> "
               "recommend; where/track/courier -> status; otherwise faq "
               "works well as the default.")

    # ----------------------------------------------------------------------
    # STEP 10.5 — Your final report (scroll up to the REPORT dict)
    # ----------------------------------------------------------------------
    say("", "STEP 10.5 — the write-up")
    labels = {"biggest_win": "Biggest win", "hardest_bug": "Hardest bug",
              "favorite_model": "Favorite model", "one_improvement":
              "With two more weeks", "advice": "Advice for the next student"}
    for key, label in labels.items():
        if len(REPORT[key].strip()) < 30:
            say(f"  ✍️  REPORT['{key}'] needs a real answer (a sentence or "
                f"three). Scroll to the REPORT dict above Assignment 10.")
            raise NotDoneYet("10.5")
    say("", "   ══════════════ FINAL REPORT — " + "FLAVORCART ML ══════════════")
    for key, label in labels.items():
        say(f"   {label}:", f"      {REPORT[key].strip()}", "")

    _mark_done(10)
    say("   " + "═" * 58, "",
        "   🏆🏆🏆  PROJECT COMPLETE  🏆🏆🏆", "",
        "   You built: a predictor, a recommender, an anomaly detector,",
        "   word embeddings, an attention model, a CNN, a language model,",
        "   a RAG chatbot, an evaluation suite, and an agent.",
        "   That is not a homework. That is a portfolio. Well done.", "")
    show_progress()


# ==========================================================================
# THE LAUNCHER — python ml2_final_project.py <1-10 | progress>
# ==========================================================================

_ASSIGNMENTS = {1: assignment_1, 2: assignment_2, 3: assignment_3,
                4: assignment_4, 5: assignment_5, 6: assignment_6,
                7: assignment_7, 8: assignment_8, 9: assignment_9,
                10: assignment_10}


def main(argv):
    if not argv or argv[0] == "progress":
        show_progress()
        if not argv:
            say("", "   Run an assignment with:  python ml2_final_project.py 1")
        return
    try:
        n = int(argv[0])
        fn = _ASSIGNMENTS[n]
    except (ValueError, KeyError):
        say(f"   '{argv[0]}' isn't an assignment. Use a number 1-10, or 'progress'.")
        return
    try:
        fn()
    except NotDoneYet:
        say("", "   ⏹  Stopped at the step above — that's normal! Edit, save,",
            f"      and run  python ml2_final_project.py {n}  again.")


if __name__ == "__main__":
    main(sys.argv[1:])
