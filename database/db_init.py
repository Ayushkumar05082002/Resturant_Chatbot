import sqlite3

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

# Basic database operations
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
    """Get all bookings or filter by date and time"""
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
