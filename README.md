# STOA Restaurant Management System

An AI-powered restaurant management system with table reservations, menu queries, and booking management capabilities.

## Features

- **AI-Powered Menu Assistant**
  - Natural language menu queries
  - Dish recommendations and details
  - Dining policy information

- **Smart Reservation System**
  - Individual table bookings
  - Group event reservations
  - Real-time availability checking
  - Automated validation and conflict prevention

- **User Interface**
  - Clean, modern web interface
  - Interactive booking calendar
  - Dynamic availability indicators
  - Mobile-responsive design

## Prerequisites

- Python 3.8+
- Tesseract OCR engine installed
- GROQ API key for AI functionality

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt


# Create .env file
GROQ_API_KEY=your_api_key_here

├── app6.py              # Main application file
├── database/            # Database management
│   └── db_init.py      # Database initialization
├── menu_images/         # Menu image storage
├── restaurant_images/   # Restaurant image storage
└── requirements.txt     # Project dependencies


## Usage

1. Start the application:
```bash
streamlit run app6.py
```

2. Initialize the knowledge base using the sidebar button

3. Navigate through the interface:
   - Home: Overview and restaurant images
   - Menu Query: AI-powered menu assistance
   - Table Reservation: Book tables and check availability
   - About Us: Restaurant information
   - Contact Us: Location and contact details

## Database Schema

```sql
CREATE TABLE reservations (
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
```

## Features in Detail

### Reservation System
- Individual and group bookings
- Automatic capacity management
- Conflict prevention
- Special requirements handling
- Email validation
- Phone number validation

### Menu System
- Image-based menu text extraction
- Natural language query processing
- Menu item recommendations
- Dietary restriction handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary and confidential.
