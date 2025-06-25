import json
import re
import os

def get_item_suggestions():
    price_file = "NEW/items_price_uom.json"
    if not os.path.exists(price_file):
        abs_path = os.path.join(os.path.dirname(__file__), "items_price_uom.json")
        if os.path.exists(abs_path):
            price_file = abs_path
    with open(price_file, "r", encoding="utf-8") as f:
        price_data = json.load(f)
    return sorted([entry["item_name"] for entry in price_data])

item_suggestions = get_item_suggestions()

def recommend_template(event_type, budget_per_head):
    """
    Recommendation Engine for Catering Templates
    
    Args:
        event_type (str): Either "Traditional" or "Party"
        budget_per_head (int): Budget in ₹ per person
    
    Returns:
        dict: Recommended template with details
    """
   
    if event_type.lower() == "traditional":
        # Traditional Event Templates
        
        if 150 <= budget_per_head <= 179:
            return {
                "template_name": "Rice & Classic Sides",
                "budget_range": "₹150–179",
                "inclusions": "1 Rice (Steamed/Bagara/Pulao) + Auto Sides (Dal, Sambar, Chutney, Pickle)",
                "items": [
                    {"name": "White Rice", "quantity": "400g"},
                    {"name": "Tomato Pappu", "quantity": "120g"},
                    {"name": "Roti Pachadi", "quantity": "25g"}
                ],
                "event_type": "Traditional"
            }
            
        elif 180 <= budget_per_head <= 219:
            return {
                "template_name": "Simple Rice",
                "budget_range": "₹180–219", 
                "inclusions": "1 Flavoured Rice (like Pulihora/Jeera) with essential accompaniments",
                "items": [
                    {"name": "Temple Style Pulihora", "quantity": "500g"},
                    {"name": "Pickle", "quantity": "40g"},
                    {"name": "Podi", "quantity": "30g"}
                ],
                "event_type": "Traditional"
            }
            
        elif 220 <= budget_per_head <= 249:
            return {
                "template_name": "Traditional Balanced Meal",
                "budget_range": "₹220–249",
                "inclusions": "1 Rice + 1 Dal/Liquid + 1 Curry + 1 Fry + 1 Pickle/Chutney + 1 Dessert",
                "items": [
                    {"name": "White Rice", "quantity": "300g"},
                    {"name": "Gutti Vankaya Curry", "quantity": "120g"},
                    {"name": "Beerakaya Pappu", "quantity": "70g"},
                    {"name": "Bhindi Fry", "quantity": "35g"},
                    {"name": "Roti Pachadi", "quantity": "20g"},
                    {"name": "Double Ka Meetha", "quantity": "100g"}
                ],
                "event_type": "Traditional"
            }
            
        elif 250 <= budget_per_head <= 299:
            return {
                "template_name": "South Indian Classic",
                "budget_range": "₹250–299",
                "inclusions": "1 Starter + 1 Biryani + 1 Flavoured Rice/Bread + 1 Curry + Dal + Fry + Dessert",
                "items": [
                    {"name": "Veg Manchuria", "quantity": "120g"},
                    {"name": "Veg Biryani", "quantity": "350g"},
                    {"name": "Rumali Roti", "quantity": "1Pcs"},
                    {"name": "Gutti Vankaya Curry", "quantity": "120g"},
                    {"name": "Cabbage Fry", "quantity": "35g"},
                    {"name": "Apricot Dessert", "quantity": "1Pcs"},
                    {"name": "Vankaya Pappu", "quantity": "70g"}
                ],
                "event_type": "Traditional"
            }
            
        elif 300 <= budget_per_head <= 349:
            return {
                "template_name": "North Indian Plate",
                "budget_range": "₹300–349",
                "inclusions": "1 Starter + 1 Biryani + 1 Flavoured Rice + 1 Bread + 2 Curries + 1 Dal + 1 Chutney + Dessert",
                "items": [
                    {"name": "Alasanda Vada", "quantity": "2Pcs"},
                    {"name": "Veg Aloo Dum Biryani", "quantity": "300g"},
                    {"name": "Chapathi", "quantity": "1Pcs"},
                    {"name": "Temple Style Pulihora", "quantity": "100g"},
                    {"name": "Rajma Curry", "quantity": "90g"},
                    {"name": "Kadai Veg Curry", "quantity": "90g"},
                    {"name": "Tomato Pappu", "quantity": "70g"},
                    {"name": "Dondakaya Chutney", "quantity": "40g"},
                    {"name": "Double Ka Meetha", "quantity": "100g"}
                ],
                "event_type": "Traditional"
            }
            
        elif 350 <= budget_per_head <= 399:
            return {
                "template_name": "Traditional Feast",
                "budget_range": "₹350–399",
                "inclusions": "1 Starter + Biryani + 2 Flavoured Rice + 1 Bread + 1 Curries + 1 Dal + 1 Liquids + 1 Chutney + 1 Dessert",
                "items": [
                    {"name": "Baby Corn 65", "quantity": "120g"},
                    {"name": "Veg Dum Aloo Biryani", "quantity": "340g"},
                    {"name": "Curd Rice", "quantity": "90g"},
                    {"name": "Sambar Rice", "quantity": "90g"},
                    {"name": "Green Peas Kofta Curry", "quantity": "120g"},
                    {"name": "Beerakaya Pappu", "quantity": "70g"},
                    {"name": "Dondakaya Chutney", "quantity": "40g"},
                    {"name": "Apricot Delight", "quantity": "1Pcs"}
                ],
                "event_type": "Traditional"
            }
            
        elif 400 <= budget_per_head <= 449:
            return {
                "template_name": "Festive Royal Combo",
                "budget_range": "₹400–449",
                "inclusions": "1 Traditional Sweet, 1 Starter, 1 Biryani, 1 Flavoured Rice, 1 Bread, 2 Curries, 1 Dal, 1 Fry, 1 Chutney, 1 Pickle, 1 Desserts",
                "items": [
                    {"name": "Gummadikaya Halwa", "quantity": "100g"},
                    {"name": "Veg 65", "quantity": "120g"},
                    {"name": "Veg Biryani", "quantity": "300g"},
                    {"name": "Sambar Rice", "quantity": "100g"},
                    {"name": "Rumali Roti", "quantity": "1Pcs"},
                    {"name": "Rajma Curry", "quantity": "90g"},
                    {"name": "Kadai Veg", "quantity": "90g"},
                    {"name": "Cabbage Fry", "quantity": "30g"},
                    {"name": "Dondakaya Chutney", "quantity": "20g"},
                    {"name": "Apricot Delight", "quantity": "1Pcs"}
                ],
                "event_type": "Traditional"
            }
            
        elif budget_per_head >= 500:
            return {
                "template_name": "Ultra Traditional Thali",
                "budget_range": "₹500+",
                "inclusions": "1 Welcome Drink, 1 Hot Snack, 1 Traditional Sweet, 1 Starter, 1 Biryani, 1 Flavoured Rice, 1 Bread, 2 Curries, 1 Dal, 1 Fry, 1 Chutney, 1 Pickle, 1 Desserts",
                "items": [
                    {"name": "Rose Milk", "quantity": "100ml"},
                    {"name": "Jalapeno Cheese Pops", "quantity": "120g"},
                    {"name": "Hara Bara Kebab", "quantity": "2Pcs"},
                    {"name": "Gummadikaya Halwa", "quantity": "100g"},
                    {"name": "Veg Dum Biryani", "quantity": "300g"},
                    {"name": "Temple Style Pulihora", "quantity": "100g"},
                    {"name": "Ghee Phulka", "quantity": "2Pcs"},
                    {"name": "Veg Pakora", "quantity": "40g"},
                    {"name": "Green Peas Curry", "quantity": "90g"},
                    {"name": "Dum Aloo Curry", "quantity": "90g"},
                    {"name": "Dosakaya Pachadi", "quantity": "30g"},
                    {"name": "Pickle", "quantity": "10g"},
                    {"name": "Dosakaya Pappu", "quantity": "60g"},
                    {"name": "Double Ka Meetha", "quantity": "100g"}
                ],
                "event_type": "Traditional"
            }
            
        else:
            return {
                "error": "Budget below minimum range for Traditional events (₹150+)",
                "suggestion": "Please increase budget to at least ₹150 per head"
            }
    
    else:  # Event type is "Party"
        # Party/Premium Event Templates
        
        if 129 <= budget_per_head <= 149:
            return {
                "template_name": "Just Biryani",
                "budget_range": "₹129–149",
                "inclusions": "1 Biryani + Inbuilt Accompaniments",
                "items": [
                    {"name": "Veg Biryani", "quantity": "500g"},
                    {"name": "Raita", "quantity": "50g"}
                ],
                "event_type": "Party"
            }
            
        elif 150 <= budget_per_head <= 179:
            return {
                "template_name": "Biryani with Dessert",
                "budget_range": "₹150–179",
                "inclusions": "1 Biryani + 1 Dessert",
                "items": [
                    {"name": "Chicken Fry Piece Biryani", "quantity": "490g"},
                    {"name": "Baked Gulab Jamun", "quantity": "1Pcs"}
                ],
                "event_type": "Party"
            }
            
        elif 180 <= budget_per_head <= 219:
            return {
                "template_name": "Starter Biryani Delight",
                "budget_range": "₹180–219",
                "inclusions": "1 Starter + 1 Biryani + 1 Dessert",
                "items": [
                    {"name": "Chicken 65", "quantity": "120g"},
                    {"name": "Chicken Dum Biryani", "quantity": "400g"},
                    {"name": "Rasamalai", "quantity": "100g"}
                ],
                "event_type": "Party"
            }
            
        elif 220 <= budget_per_head <= 249:
            return {
                "template_name": "Feast Lite",
                "budget_range": "₹220–249",
                "inclusions": "1 Starter + 1 Biryani + 1 Bread + 1 Curry + 1 Dessert",
                "items": [
                    {"name": "Cheese Corn Balls", "quantity": "100g"},
                    {"name": "Chicken Dum Biryani", "quantity": "290g"},
                    {"name": "Chapati", "quantity": "1Pcs"},
                    {"name": "Baingan Masala", "quantity": "100g"},
                    {"name": "Double Ka Meetha", "quantity": "100g"}
                ],
                "event_type": "Party"
            }
            
        elif 250 <= budget_per_head <= 299:
            return {
                "template_name": "Double Starter Combo",
                "budget_range": "₹250–299",
                "inclusions": "2 Starters + 1 Biryani + 1 Bread + 1 Curry + 1 Dessert",
                "items": [
                    {"name": "Cheese Corn Balls", "quantity": "90g"},
                    {"name": "Chicken 65", "quantity": "90g"},
                    {"name": "Chicken Dum Biryani", "quantity": "260g"},
                    {"name": "Chapati", "quantity": "1Pcs"},
                    {"name": "Baingan Masala", "quantity": "90g"},
                    {"name": "Double Ka Meetha", "quantity": "90g"}
                ],
                "event_type": "Party"
            }
            
        elif 300 <= budget_per_head <= 349:
            return {
                "template_name": "3 Starter Party Box",
                "budget_range": "₹300–349",
                "inclusions": "3 Starters + 1 Biryani + 1 Bread + 1 Curry + 1 Dessert",
                "items": [
                    {"name": "Chilli Baby Corn", "quantity": "70g"},
                    {"name": "Cheese Fries", "quantity": "70g"},
                    {"name": "Chilli Egg", "quantity": "70g"},
                    {"name": "Chicken Dum Biryani", "quantity": "280g"},
                    {"name": "Chapati", "quantity": "1Pcs"},
                    {"name": "Dum Ka Murgh Curry", "quantity": "100g"},
                    {"name": "Tiramisu", "quantity": "1Pcs"}
                ],
                "event_type": "Party"
            }
            
        elif 350 <= budget_per_head <= 399:
            return {
                "template_name": "4 Starter Celebration Box",
                "budget_range": "₹350–399",
                "inclusions": "4 Starters + 1 Biryani + 1 Bread + 1 Curry + 1 Dessert",
                "items": [
                    {"name": "Kaju Chicken Pakoda", "quantity": "60g"},
                    {"name": "Malai Paneer Tikka", "quantity": "60g"},
                    {"name": "Chicken Galouti Kebab", "quantity": "1Pcs"},
                    {"name": "Baby Corn Manchurian", "quantity": "60g"},
                    {"name": "Chicken Dum Biryani", "quantity": "250g"},
                    {"name": "Chapati", "quantity": "1Pcs"},
                    {"name": "Kadai Chicken", "quantity": "80g"},
                    {"name": "Double Ka Meetha", "quantity": "100g"}
                ],
                "event_type": "Party"
            }
            
        elif 400 <= budget_per_head <= 449:
            return {
                "template_name": "6 Starter Deluxe Box",
                "budget_range": "₹400–449",
                "inclusions": "6 Starters + 1 Biryani + 1 Dessert",
                "items": [
                    {"name": "Beetroot Galouti Kebab", "quantity": "2Pcs"},
                    {"name": "Pachi Mirchi Kodi Kebab", "quantity": "50g"},
                    {"name": "Palak Pakoda", "quantity": "50g"},
                    {"name": "Paneer Majestic", "quantity": "50g"},
                    {"name": "Peri Peri Chicken Wings", "quantity": "50g"},
                    {"name": "Peri Peri Fries", "quantity": "50g"},
                    {"name": "Chicken Dum Biryani", "quantity": "300g"},
                    {"name": "Tiramisu", "quantity": "1Pcs"}
                ],
                "event_type": "Party"
            }
            
        elif 450 <= budget_per_head <= 499:
            return {
                "template_name": "Royal Grand Combo",
                "budget_range": "₹450–499",
                "inclusions": "6 Starters + 1 Biryani + 1 Flavoured Rice + 2 Desserts",
                "items": [
                    {"name": "Chicken 65", "quantity": "50g"},
                    {"name": "Veg Manchurian Dry", "quantity": "50g"},
                    {"name": "Veg Nuggets", "quantity": "1Pcs"},
                    {"name": "Veg Pockets", "quantity": "1Pcs"},
                    {"name": "Chili Garlic Mushroom", "quantity": "50g"},
                    {"name": "Paneer Tikka Kebab", "quantity": "50g"},
                    {"name": "Chicken Zafrani Biryani", "quantity": "280g"},
                    {"name": "Jeera Rice", "quantity": "100g"},
                    {"name": "Apricot Delight", "quantity": "1Pcs"},
                    {"name": "Double Ka Meetha", "quantity": "100g"}
                ],
                "event_type": "Party"
            }
            
        elif budget_per_head >= 500:
            return {
                "template_name": "Mega Celebration",
                "budget_range": "₹500+",
                "inclusions": "10 Starters + 1 Biryani + 1 Bread + 2 Curries + 2 Flavoured Rices + 2 Desserts",
                "items": [
                    {"name": "Honey Chilli Chicken", "quantity": "50g"},
                    {"name": "Honey Chilli Potato", "quantity": "50g"},
                    {"name": "Honey Garlic Chicken Bites", "quantity": "50g"},
                    {"name": "Chicken Joojeh Kebab", "quantity": "50g"},
                    {"name": "Kaju Chicken Pakoda", "quantity": "50g"},
                    {"name": "Malai Paneer Tikka", "quantity": "50g"},
                    {"name": "Paneer Tikka Kebab", "quantity": "50g"},
                    {"name": "Chilli Prawns", "quantity": "50g"},
                    {"name": "Chilli Chicken Lollipop", "quantity": "1Pcs"},
                    {"name": "Crispy Chicken Wings", "quantity": "50g"},
                    {"name": "Chicken Dum Biryani", "quantity": "260g"},
                    {"name": "Ghee Phulka", "quantity": "2Pcs"},
                    {"name": "Dum Ka Murgh", "quantity": "90g"},
                    {"name": "Kadai Veg", "quantity": "80g"},
                    {"name": "Sambar Rice", "quantity": "90g"},
                    {"name": "Curd Rice", "quantity": "90g"},
                    {"name": "Apricot Delight", "quantity": "1Pcs"},
                    {"name": "Rasamalai", "quantity": "100g"}
                ],
                "event_type": "Party"
            }
            
        else:
            return {
                "error": "Budget below minimum range for Party events (₹129+)",
                "suggestion": "Please increase budget to at least ₹129 per head"
            }


def parse_quantity(qty):
    # Extract numeric value and unit
    match = re.match(r"([\d.]+)\s*([a-zA-Z]*)", qty.strip())
    if not match:
        return None, None
    value, unit = match.groups()
    return float(value), unit.lower()

def calculate_template_price(template, price_file="NEW/items_price_uom.json", use_peak=False):
    # Try both relative and absolute paths
    if not os.path.exists(price_file):
        abs_path = os.path.join(os.path.dirname(__file__), "items_price_uom.json")
        if os.path.exists(abs_path):
            price_file = abs_path
        else:
            raise FileNotFoundError(f"Could not find {price_file} or {abs_path}")
    with open(price_file, "r", encoding="utf-8") as f:
        price_data = json.load(f)

    def get_price_info(item_name):
        for entry in price_data:
            if entry["item_name"].lower() == item_name.lower():
                return entry
        return None

    total_price = 0
    for item in template["items"]:
        name = item["name"]
        qty = item["quantity"]
        price_info = get_price_info(name)
        if not price_info or not qty:
            continue

        qty_num, qty_unit = parse_quantity(qty)
        if qty_num is None:
            continue

        uom = price_info["uom"].lower()
        price_per_unit = price_info["cmp_peak_price"] if use_peak else price_info["cmp_base_price"]

        if uom == "kg":
            if qty_unit == "g":
                qty_num = qty_num / 1000
            elif qty_unit == "kg":
                pass  # already in kg
            else:
                continue  # skip if not parsable
        elif uom == "pcs":
            pass  # already in pieces
        elif uom == "ml":
            if qty_unit == "ml":
                qty_num = qty_num / 1000
            elif qty_unit == "l":
                pass  # already in litres
            else:
                continue
        else:
            continue  # skip unknown units

        total_price += qty_num * price_per_unit

    return round(total_price, 2)

# Streamlit UI
try:
    import streamlit as st
    st.set_page_config(layout="centered")
    st.title("Find My Plate🍽️")
    with st.sidebar:
        st.header("Options")
        event_type = st.radio("Event Type", ["Traditional", "Party"], index=0)
        if event_type == "Traditional":
            min_budget, max_budget = 150, 700
        else:
            min_budget, max_budget = 129, 700
        budget_per_head = st.slider("Budget per head (₹)", min_value=min_budget, max_value=max_budget, value=min_budget, step=1)
        use_peak = st.checkbox("Use peak price?", value=True)
        submit = st.button("Get Recommendation")
    if submit:
        result = recommend_template(event_type, budget_per_head)
        st.session_state['last_recommendation'] = {
            'result': result,
            'event_type': event_type,
            'budget_per_head': budget_per_head,
            'use_peak': use_peak
        }

    # Layout: Recommendation card (left), Add-on calculator (right)
    col_left, col_right = st.columns([2.2, 1.3], gap="medium")
    with col_left:
        # Always display the last recommendation if it exists
        show_recommendation = 'last_recommendation' in st.session_state
        if show_recommendation:
            rec = st.session_state['last_recommendation']
            result = rec['result']
            use_peak = rec['use_peak']
            st.markdown("<div style='display: flex; justify-content: flex-start;'>", unsafe_allow_html=True)
            if 'error' in result:
                st.error(result['error'])
                st.info(result['suggestion'])
            else:
                st.markdown(
                    f"""
                    <div style='background: #fff8f0; border-radius: 12px; padding: 2em; box-shadow: 0 2px 8px #eee; max-width: 600px; margin: auto;'>
                        <h2 style='color: #FF6F61;'>{result['template_name']}</h2>
                        <p><b>Budget Range:</b> {result['budget_range']}<br>
                        <b>Event Type:</b> {result['event_type']}<br>
                        <b>Inclusions:</b> {result['inclusions']}</p>
                        <hr>
                        <h4>Menu Items</h4>
                        <ul style='padding-left: 1.2em;'>
                            {''.join([f'<li style=\'margin-bottom: 0.5em;\'>{item['name']} <span style=\'color: #888;\'>({item['quantity']})</span></li>' for item in result['items']])}
                        </ul>
                        <hr>
                        <h3 style='color: #388e3c;'>Estimated Total Price: ₹{calculate_template_price(result, use_peak=use_peak)}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        # Only show add-on section if a recommendation is displayed
        if show_recommendation:
            st.markdown("""
                <style>
                .addon-container {
                    background: #f8f9fa;
                    border-radius: 16px;
                    padding: 2.5em 2em 2em 2em;
                    box-shadow: 0 2px 16px #eee;
                    margin-top: 2em;
                    min-width: 520px;
                    max-width: 700px;
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                }
                .addon-form-row input, .addon-form-row select {
                    width: 100% !important;
                    min-width: 180px !important;
                    max-width: 260px !important;
                    padding: 0.7em 1em !important;
                    margin-bottom: 0.7em !important;
                    font-size: 1.1em !important;
                }
                </style>
                <div class='addon-container'>
            """, unsafe_allow_html=True)
            st.header("➕ Add-on", divider="rainbow")
            if 'addon_items' not in st.session_state:
                st.session_state['addon_items'] = []
            with st.form("addon_form"):
                st.markdown("<div class='addon-form-row' style='display: flex; flex-direction: row; gap: 2em;'>", unsafe_allow_html=True)
                # Use selectbox for item name with suggestions, first option is empty
                addon_name = st.selectbox("Item name", options=[""] + item_suggestions, key="addon_name")
                addon_qty = ""
                if addon_name:
                    addon_qty = st.text_input("Quantity (e.g., 100g, 2Pcs, 200ml)", key="addon_qty")
                st.markdown("</div>", unsafe_allow_html=True)
                add_item = st.form_submit_button("Add Item")
            if add_item:
                if not addon_name or not addon_qty:
                    st.warning("Please enter both item name and quantity.")
                else:
                    st.session_state['addon_items'].append({"name": addon_name, "quantity": addon_qty})
            if st.session_state['addon_items']:
                st.markdown("<b>Current Add-on Items:</b>", unsafe_allow_html=True)
                st.markdown(
                    "<ul style='padding-left: 1.2em;'>" +
                    ''.join([f"<li style='margin-bottom: 0.5em;'>{item['name']} <span style='color: #888;'>({item['quantity']})</span></li>" for item in st.session_state['addon_items']]) +
                    "</ul>",
                    unsafe_allow_html=True
                )
                if st.button("Clear Add-on List"):
                    st.session_state['addon_items'] = []
                # Calculate total add-on price
                try:
                    addon_template = {"items": st.session_state['addon_items']}
                    addon_price = calculate_template_price(addon_template, use_peak=True)
                    st.success(f"Total Add-on Price: ₹{addon_price}")
                except Exception as e:
                    st.error(f"Could not calculate price: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
except ImportError:
    pass

# Example Usage:
if __name__ == "__main__":
    print("=== Catering Recommendation Engine ===")
    event_type = input("Enter event type (Traditional/Party): ").strip()
    budget_str = input("Enter budget per head (in ₹): ").strip()
    use_peak_str = input("Use peak price? (yes/no): ").strip().lower()
    use_peak = use_peak_str in ["yes", "y"]
    try:
        budget_per_head = int(budget_str)
    except ValueError:
        print("Invalid budget. Please enter a number.")
        exit(1)
    result = recommend_template(event_type, budget_per_head)
    if 'error' in result:
        print(f"Error: {result['error']}")
        print(f"Suggestion: {result['suggestion']}")
    else:
        print(f"\nRecommended Template: {result['template_name']}")
        print(f"Budget Range: {result['budget_range']}")
        print(f"Event Type: {result['event_type']}")
        print(f"Inclusions: {result['inclusions']}")
        print("Items:")
        for item in result['items']:
            print(f"  - {item['name']} ({item['quantity']})")
        total = calculate_template_price(result, use_peak=use_peak)
        print(f"Estimated Total Price: ₹{total}")
