import os
import instructor
import streamlit as st
import groq
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema
import asyncio  
from datetime import datetime, timedelta  
import re  

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import sqlite3

# --- Initialize Database ---
def init_db():
    """Initialize the SQLite database with tables"""
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reservations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        time TEXT,
        guests INTEGER,
        name TEXT,
        email TEXT,
        phone TEXT,
        special_requests TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

# --- Database Functions ---
def add_booking(date, time, guests, name='', email='', phone='', special_requests=''):
    """Add a new booking to the database"""
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO reservations (date, time, guests, name, email, phone, special_requests) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (date, time, guests, name, email, phone, special_requests)
    )
    
    conn.commit()
    conn.close()
    return True

def get_bookings(date=None, time=None):
    """Get all bookings or filter by date and time with booking details"""
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    
    query = '''SELECT date, time, guests, name, email, special_requests 
              FROM reservations'''
    params = []
    
    if date and time:
        query += ' WHERE date = ? AND time = ?'
        params = [date, time]
    elif date:
        query += ' WHERE date = ?'
        params = [date]
    
    query += ' ORDER BY time'
    cursor.execute(query, params)
    bookings = cursor.fetchall()
    conn.close()
    
    return bookings

def check_availability(date, time, party_size):
    """Check if a reservation can be made for given date, time and party size"""
    max_capacity = 50  # Get from configuration
    
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    
    # Get total guests for that time slot
    cursor.execute('SELECT SUM(guests) FROM reservations WHERE date = ? AND time = ?', (date, time))
    result = cursor.fetchone()
    current_guests = result[0] if result[0] else 0
    
    conn.close()
    
    # Check if adding party_size would exceed capacity
    return (current_guests + party_size <= max_capacity, max_capacity - current_guests)

# Call the function to initialize the database
init_db()

print("Database 'restaurant_bookings.db' has been created successfully.")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API Key
if not GROQ_API_KEY:
    raise ValueError("❌ Error: GROQ_API_KEY is not set in .env file")

# Initialize Groq client
groq_client = groq.Client(api_key=GROQ_API_KEY)

# Wrap Groq's client with Instructor
client = instructor.from_groq(groq_client)

# Function to extract text from menu images
def extract_text_from_images(image_directory):
    extracted_text = []
    for image_file in os.listdir(image_directory):
        if image_file.endswith((".jpeg", ".jpg", ".png")):
            image_path = os.path.join(image_directory, image_file)
            text = pytesseract.image_to_string(Image.open(image_path))
            extracted_text.append(text)
    
    return "\n".join(extracted_text) if extracted_text else None

# Sidebar
st.sidebar.image("1.png", use_container_width=True)
st.sidebar.title("Navigation")
st.sidebar.markdown("👋 Welcome to the AI-powered Restaurant Assistant!")

# Main Header
# st.image("2.jpg", use_container_width=True)
# st.title("🍽️ Welcome To STOA")
# st.markdown("**Ask anything about our restaurant, menu, and reservations!**")

# st.subheader("🏡 Our Dining Area")
# dining_area_path = "restaurant_images/dining_area.jpg"

# if os.path.exists(dining_area_path):
#     st.image(dining_area_path, caption="A glimpse of our cozy dining area!", use_container_width=True)
# else:
#     st.warning("Dining area image not found.")

# Initialize Knowledge Base
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
    st.session_state.vectors = None
    st.session_state.chat_history = []

if st.sidebar.button("🔄 Initialize Knowledge Base"):
    with st.spinner("Processing images..."):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            image_directory = "menu_images"
            all_text = []

            for image_file in os.listdir(image_directory):
                if image_file.endswith((".jpeg", ".jpg")):
                    image_path = os.path.join(image_directory, image_file)
                    with Image.open(image_path) as img:
                        text = pytesseract.image_to_string(img)
                        all_text.append(text)

            if not all_text:
                st.sidebar.error("No text extracted from images. Please check the files and retry.")
            else:
                combined_text = "\n".join(all_text)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_text(combined_text)
                vectors = FAISS.from_texts(final_documents, embeddings)
                st.session_state.vectors = vectors
                st.session_state.vector_ready = True
                st.sidebar.success("Knowledge Base Initialized!")
        except Exception as e:
            st.sidebar.error(f"Error initializing: {e}")

# Load and Initialize Menu Data
# menu_directory = "menu_images"  # Folder containing menu images
# if "menu_vector_ready" not in st.session_state:
#     menu_text = extract_text_from_images(menu_directory)
    
#     if menu_text:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#         menu_documents = text_splitter.split_text(menu_text)
#         st.session_state.menu_vectors = FAISS.from_texts(menu_documents, embeddings)
#         st.session_state.menu_vector_ready = True
#         st.sidebar.success("✅ Menu Data Initialized!")

# Function to retrieve menu information
def get_menu_response(query):
    if not st.session_state.get("menu_vector_ready", False):
        return "⚠️ Menu data is not initialized."
    
    menu_vectors = st.session_state.menu_vectors
    docs = menu_vectors.similarity_search(query, k=3)  # Retrieve top matching menu items
    return "\n".join([doc.page_content for doc in docs]) if docs else "No matching menu items found."

# Create Menu Agent with structured responses
menu_agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="llama3-70b-8192",
        temperature=0.1,  # Lower temperature for consistency
        system_prompt=f"""
        You are a restaurant assistant helping customers with menu-related queries.

        
        When answering queries:
        - If the customer asks **"What is in the menu?"**, list **all available dishes** clearly.
        - If they ask about a specific dish, provide details from the extracted menu.
        - If the menu data is missing or unclear, politely inform the customer.
        - Do NOT repeat the user's question.
        """
    )
)

# Create Reservation Agent
reservation_agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="llama3-8b-8192",
        temperature=0.1,
        system_prompt="You are a restaurant booking assistant. Help customers with table reservations, availability, and booking details."
    )
)

# Function to handle reservation queries
def handle_reservation_query(user_input):
    """Process reservation queries and interact with the database."""
    try:
        # Extract relevant information from user input
        number_of_people = int(input("How many people are you booking for? "))
        reservation_date = input("Please provide the date for the reservation (YYYY-MM-DD): ")
        reservation_time = input("Please provide the time for the reservation (HH:MM): ")

        # Combine date and time
        reservation_datetime = f"{reservation_date} {reservation_time}"

        # Add booking to the database
        add_booking(reservation_datetime, number_of_people)

        print(f"✅ Reservation for {number_of_people} people on {reservation_datetime} has been successfully booked.")
    
    except Exception as e:
        print(f"❌ Error processing reservation: {e}")

#Function to handle user queries
def handle_query(user_input):
    """Determine which agent to use based on the query and return AI response."""
    try:
        if any(word in user_input.lower() for word in ["menu", "food", "dish", "hours", "best-selling","pizza","pasta","burger","sandwich","salad","dessert","beverage","drink","group-reservations","restaurant-capacity","dining-guidelines"]):
            if not st.session_state.get("vector_ready", False):
                print("⚠️ Knowledge base is not initialized.")
                return
            similar_docs = st.session_state.vectors.similarity_search(user_input, k=1)
            retrieved_context = "\n".join(
                [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in similar_docs]
            )
            enhanced_question = f"Context:\n{retrieved_context}\n\nQuestion:\n{user_input}"
            response = menu_agent.run(BaseAgentInputSchema(chat_message=enhanced_question))
            print(response.chat_message)
        
        elif any(word in user_input.lower() for word in ["reserve", "book", "table", "availability"]):
            handle_reservation_query(user_input)
        
        else:
            print("🤖 I can assist with menu-related queries and reservations. Please ask accordingly!")
    
    except Exception as e:
        print(f"❌ Error processing request: {e}")

# Function to check availability
def check_availability(date, time, number_of_people):
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT SUM(number_of_people) FROM bookings
        WHERE date = ? AND time = ?
    ''', (date, time))
    total_people = cursor.fetchone()[0]
    conn.close()
    
    if total_people is None:
        total_people = 0
    
    # Assuming the restaurant can accommodate 50 people at a time
    max_capacity = 50
    available_slots = max_capacity - total_people
    
    return available_slots >= number_of_people, available_slots

# Function to update availability after booking
def update_availability(date, time, number_of_people):
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT SUM(number_of_people) FROM bookings
        WHERE date = ? AND time = ?
    ''', (date, time))
    total_people = cursor.fetchone()[0]
    
    if total_people is None:
        total_people = 0
    
    total_people += number_of_people
    
    # Assuming the restaurant can accommodate 50 people at a time
    max_capacity = 50
    available_slots = max_capacity - total_people
    
    conn.commit()
    conn.close()
    
    return available_slots >= 0, available_slots

# Function to check availability based on empty tables
def check_table_availability(date, time, number_of_people, table_capacity=4, total_tables=12):
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT SUM(number_of_people) FROM bookings
        WHERE date = ? AND time = ?
    ''', (date, time))
    total_people = cursor.fetchone()[0]
    conn.close()
    
    if total_people is None:
        total_people = 0
    
    # Calculate tables needed for the new reservation
    tables_needed = (number_of_people + table_capacity - 1) // table_capacity
    
    # Calculate tables needed for existing reservations
    tables_in_use = (total_people + table_capacity - 1) // table_capacity
    
    # Calculate available tables
    available_tables = total_tables - tables_in_use
    
    return available_tables >= tables_needed, available_tables

# Function to update availability after booking
def update_table_availability(date, time, number_of_people, table_capacity=4):
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT SUM(number_of_people) FROM bookings
        WHERE date = ? AND time = ?
    ''', (date, time))
    total_people = cursor.fetchone()[0]
    
    if total_people is None:
        total_people = 0
    
    total_people += number_of_people
    
    conn.commit()
    conn.close()
    
    return total_people

# Streamlit UI for reservations
def reservation_ui():
    st.header("Table Reservation")
    with st.form("reservation_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", "Enter your name")
            email = st.text_input("Email")
            phone = st.text_input("Contact Number", "Enter your contact number")
            number_of_people = st.number_input("Number of People", min_value=1, step=1)
        
        with col2:
            reservation_date = st.date_input("Reservation Date")
            reservation_time = st.time_input("Reservation Time")

        check_availability_button = st.form_submit_button("Check Availability")
        book_table_button = st.form_submit_button("Book Table")

        if check_availability_button:
            is_available, available_tables = check_table_availability(reservation_date, reservation_time, number_of_people)
            if is_available:
                st.success(f"Tables available for {number_of_people} people on {reservation_date} at {reservation_time}.")
            else:
                st.error(f"Only {available_tables} tables available for {reservation_date} at {reservation_time}.")

        if book_table_button:
            is_available, _ = check_table_availability(reservation_date, reservation_time, number_of_people)
            if is_available:
                reservation_datetime = f"{reservation_date} {reservation_time}"
                # Update this line to match the add_booking parameters
                add_booking(
                    date=reservation_date,
                    time=reservation_time,
                    guests=number_of_people,
                    name=name,
                    email=email,
                    phone=phone
                )
                update_table_availability(reservation_date, reservation_time, number_of_people)
                st.success(f"Reservation for {number_of_people} people on {reservation_datetime} has been successfully booked.")
            else:
                st.error("Selected time slot is not available. Please choose a different time.")

# Function to process mass bookings for group events
def process_mass_booking(selected_dates, time_slot, number_of_people, group_name, contact_person, contact_email, event_type, special_requirements):
    """Process multiple bookings for group events"""
    successful_dates = []
    failed_dates = []
    
    try:
        for date in selected_dates:
            # Check availability first
            is_available, _ = check_availability(date, time_slot, number_of_people)
            
            if is_available:
                # Add booking to database
                add_booking(
                    name=group_name,
                    contact_number=contact_email,
                    date=date,
                    time=time_slot,
                    number_of_people=number_of_people,
                    status=f"Event Type: {event_type}. {special_requirements}"
                )
                update_availability(date, time_slot, number_of_people)
                successful_dates.append(date)
            else:
                failed_dates.append(date)
                
        return successful_dates, failed_dates
    except Exception as e:
        st.error(f"Error processing bookings: {str(e)}")
        return successful_dates, failed_dates

# Function to process reservation form submission
def process_reservation_submission(submit_button, name, email, phone, date, time, guests, special_requests):
    """Process the reservation form submission with validation"""
    if submit_button:
        # Validate all required fields
        validation_errors = []
        
        if not name or len(name.strip()) < 2:
            validation_errors.append("Please enter a valid name (minimum 2 characters)")
        
        if not validate_email(email):
            validation_errors.append("Please enter a valid email address")
        
        if not validate_phone(phone):
            validation_errors.append("Please enter a valid 10-digit phone number")
        
        if date < datetime.now().date():
            validation_errors.append("Please select a future date")

        # Show all validation errors if any
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return

        # Check for existing booking
        existing_booking = check_existing_booking(name, email, date, time)
        if existing_booking:
            st.error("You already have a booking for this date and time!")
            return
            
        # If all validations pass, proceed with booking
        date_str = date.strftime('%Y-%m-%d')
        is_available, seats_left = check_availability(date_str, time, guests)
        
        if is_available:
            success = add_booking(
                date=date_str,
                time=time,
                guests=guests,
                name=name,
                email=email,
                phone=phone,
                special_requests=special_requests
            )
            if success:
                st.success(f"Reservation confirmed for {name} on {date_str} at {time} for {guests} guests.")
            else:
                st.error("Failed to process reservation. Please try again.")
        else:
            st.error(f"Sorry, we don't have enough space for {guests} guests at {time}. We have {seats_left} seats left at that time.")

def check_existing_booking(name, email, date, time):
    """Check if user already has a booking for the given date and time"""
    conn = sqlite3.connect('restaurant_bookings.db')
    cursor = conn.cursor()
    date_str = date.strftime('%Y-%m-%d')
    
    cursor.execute('''
        SELECT * FROM reservations 
        WHERE (name = ? OR email = ?) AND date = ? AND time = ?
    ''', (name, email, date_str, time))
    
    booking = cursor.fetchone()
    conn.close()
    return booking

def validate_phone(phone):
    """Validate phone number - must be 10 digits"""
    return bool(re.match(r'^\d{10}$', phone.strip()))

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

# Function to render the new reservation form
def render_new_reservation_form():
    """Render the new reservation form with validation hints"""
    st.markdown('<div class="reservation-card">', unsafe_allow_html=True)
    with st.form("reservation_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", help="Enter at least 2 characters")
            email = st.text_input("Email", help="Enter a valid email address")
            phone = st.text_input("Phone Number", help="Enter 10 digits only")
        
        with col2:
            min_date = datetime.now().date()
            date = st.date_input("Reservation Date", min_value=min_date)
            time = st.selectbox("Time", ["5:00 PM", "5:30 PM", "6:00 PM", "6:30 PM", "7:00 PM", "7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM"])
            guests = st.number_input("Number of Guests", min_value=1, max_value=20, value=2)
        
        special_requests = st.text_area("Special Requests (Optional)")
        submit_button = st.form_submit_button("Reserve Table")
        
        process_reservation_submission(submit_button, name, email, phone, date, time, guests, special_requests)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to render the availability checker
def render_availability_checker():
    """Render the availability checker"""
    st.markdown('<div class="reservation-card">', unsafe_allow_html=True)
    st.subheader("Check Table Availability")
    check_date = st.date_input("Select Date", key="check_date")
    
    # Get bookings for the selected date
    date_str = check_date.strftime('%Y-%m-%d')
    existing_bookings = get_bookings(date=date_str)
    
    # Calculate total guests for the day
    total_guests = sum(int(booking[2]) for booking in existing_bookings)
    max_capacity = 50  # Assuming max capacity is 50
    
    # Display availability stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reserved Seats", total_guests)
    with col2:
        st.metric("Total Capacity", max_capacity)
        
    # Show availability by time slot with booking details
    display_time_slot_availability(existing_bookings, max_capacity)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to display availability for each time slot
def display_time_slot_availability(existing_bookings, max_capacity):
    """Display availability for each time slot with booking details"""
    st.subheader("Availability by Time")
    times = ["5:00 PM", "5:30 PM", "6:00 PM", "6:30 PM", "7:00 PM", "7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM"]
    
    for time_slot in times:
        # Get bookings for this specific time slot
        time_bookings = [b for b in existing_bookings if b[1] == time_slot]
        time_guests = sum(int(b[2]) for b in time_bookings) if time_bookings else 0
        seats_left = max_capacity - time_guests
        
        # Create expandable section for each time slot
        with st.expander(f"🕒 {time_slot} - {get_availability_status(seats_left)}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Available Seats", seats_left)
                st.metric("Booked Seats", time_guests)
            
            with col2:
                if time_bookings:
                    st.markdown("### Current Bookings:")
                    for booking in time_bookings:
                        st.markdown(f"""
                        - **Party**: {booking[2]} guests
                        - **Name**: {booking[3]}
                        - **Notes**: {booking[5] if booking[5] else 'None'}
                        ---
                        """)
                else:
                    st.info("No bookings for this time slot")

def get_availability_status(seats_left):
    """Get formatted availability status with color indicators"""
    if seats_left >= 10:
        return "🟢 Available"
    elif seats_left > 0:
        return f"🟡 Limited ({seats_left} seats)"
    else:
        return "🔴 Fully Booked"

# Function to render the group booking form
def render_group_booking_form():
    """Render the group booking form"""
    st.markdown('<div class="reservation-card">', unsafe_allow_html=True)
    st.subheader("Book Multiple Dates")
    
    with st.form("group_booking_form"):
        col1, col2 = st.columns(2)
        with col1:
            group_name = st.text_input("Group/Event Name")
            contact_person = st.text_input("Contact Person")
            contact_email = st.text_input("Contact Email")
        
        with col2:
            number_of_people = st.number_input("Number of People per Reservation", 
                                            min_value=1, max_value=50, value=10)
            event_type = st.selectbox("Event Type", 
                                    ["Corporate Meeting", "Birthday Party", "Wedding Reception", 
                                    "Anniversary", "Other"])
            time_slot = st.selectbox("Preferred Time", 
                                    ["5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM"], 
                                    key="group_time")
        
        # Calendar selector
        st.subheader("Select Dates")
        available_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 15)]
        selected_dates = st.multiselect("Select all dates needed:", available_dates)
        
        # Special requirements
        special_requirements = st.text_area("Special Requirements or Notes",
                                        placeholder="Please let us know about any dietary restrictions, room setup preferences, or other special needs.")
        
        submit_group = st.form_submit_button("📅 Request Group Booking")
        
        process_group_booking_submission(submit_group, selected_dates, group_name, contact_person, contact_email,
                                         time_slot, number_of_people, event_type, special_requirements)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to process group booking form submission
def process_group_booking_submission(submit_group, selected_dates, group_name, contact_person, contact_email,
                                    time_slot, number_of_people, event_type, special_requirements):
    """Process the group booking form submission"""
    if submit_group and selected_dates and group_name and contact_person and contact_email:
        successful_dates, failed_dates = [], []
        
        for date in selected_dates:
            is_available, available_tables = check_table_availability(date, time_slot, number_of_people)
            if is_available:
                # Update this line to match the add_booking parameters
                add_booking(
                    date=date,
                    time=time_slot,
                    guests=number_of_people,
                    name=group_name,
                    email=contact_email,
                    phone=contact_person,
                    special_requests=f"Event Type: {event_type}. {special_requirements}"
                )
                update_table_availability(date, time_slot, number_of_people)
                successful_dates.append(date)
            else:
                failed_dates.append(date)
        
        if successful_dates:
            st.success(f"Successfully booked: {', '.join(successful_dates)}")
        
        if failed_dates:
            st.error(f"No availability for these dates: {', '.join(failed_dates)}")
    elif submit_group:
        st.warning("Please fill out all required fields and select at least one date.")

# Function to render the reservation page
def render_reservation_page():
    """Render the Reservations page content"""
    st.title("Make a Reservation", anchor="reservation-section") 
    
    # Reservation Tabs
    res_tab1, res_tab2 = st.tabs(["New Reservation", "Check Availability"])
    
    # New Reservation Tab
    with res_tab1:
        render_new_reservation_form()

    # Check Availability Tab
    with res_tab2:
        render_availability_checker()

    # Group & Event Reservations
    with st.expander("Group & Event Reservations"):
        render_group_booking_form()

# Streamlit UI for menu queries
def menu_query_ui():
    st.header("How can we assist you today?")
    user_input = st.text_input("Ask about the menu or restaurant details")

    if st.button("Submit"):
        if any(word in user_input.lower() for word in ["menu", "food", "dish", "hours", "best-selling","pizza","pasta","burger","sandwich","salad","dessert","beverage","liquor","opening-hours","drink","group-reservations","restaurant-capacity","dining-guidelines","pet","smoking","takeaway","delivery","cuisine","vegan","vegetarian","gluten-free","allergy","reservation","booking","table","availability"]):
            if not st.session_state.get("vector_ready", False):
                st.warning("⚠️ Knowledge base is not initialized.")
                return
            similar_docs = st.session_state.vectors.similarity_search(user_input, k=1)
            retrieved_context = "\n".join(
                [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in similar_docs]
            )
            enhanced_question = f"Context:\n{retrieved_context}\n\nQuestion:\n{user_input}"
            response = menu_agent.run(BaseAgentInputSchema(chat_message=enhanced_question))
            st.write(response.chat_message)
        else:
            st.warning("Please ask about the menu or restaurant details.")

# Streamlit UI for About section
def about_ui():
    st.header("About Us")
    st.markdown("""
    <style>
    .about-text {
        font-size: 18px;
        line-height: 1.6;
    }
    </style>
    <div class="about-text">
        Welcome to STOA, your premier dining destination! Our restaurant offers a cozy dining area, a diverse menu, and exceptional service. 
        Whether you're here for a casual meal or a special occasion, we strive to make your experience memorable.
    </div>
    """, unsafe_allow_html=True)

# Streamlit UI for Contact Us section
def contact_us_ui():
    st.header("Contact Us")
    st.markdown("""
    <style>
    .contact-text {
        font-size: 18px;
        line-height: 1.6;
    }
    </style>
    <div class="contact-text">
        <p><strong>Address:</strong> Dehmi kalan near Manipal University JAIPUR</p>
        <p><strong>Phone:</strong> (123) 456-7890</p>
        <p><strong>Email:</strong> contact@stoa.com</p>
    </div>
    """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .nav-bar {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .nav-bar a {
        margin: 0 15px;
        text-decoration: none;
        font-size: 18px;
        color: #007bff;
    }
    .nav-bar a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🍽️ Welcome To STOA</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-bar">
        <a href="#home" onclick="window.location.href='#home';">Home</a>
        <a href="#about-us" onclick="window.location.href='#about-us';">About Us</a>
        <a href="#menu-query" onclick="window.location.href='#menu-query';">Menu Query</a>
        <a href="#table-reservation" onclick="window.location.href='#table-reservation';">Table Reservation</a>
        <a href="#contact-us" onclick="window.location.href='#contact-us';">Contact Us</a>
    </div>
    """, unsafe_allow_html=True)

    option = st.selectbox("", ["Home", "About Us", "Query", "Table Reservation", "Contact Us"], key="nav_select")

    if option == "Home":
        st.image("2.jpg", use_container_width=True)
        st.markdown("**Ask anything about our restaurant, menu, and reservations!**")
        st.subheader("🏡 Our Dining Area")
        dining_area_path = "restaurant_images/dining_area.jpg"
        if os.path.exists(dining_area_path):
            st.image(dining_area_path, caption="A glimpse of our cozy dining area!", use_container_width=True)
        else:
            st.warning("Dining area image not found.")
    elif option == "About Us":
        about_ui()
    elif option == "Query":
        menu_query_ui()
    elif option == "Table Reservation":
        render_reservation_page()
    elif option == "Contact Us":
        contact_us_ui()

if __name__ == "__main__":
    main()

# --- End of app6.py ---