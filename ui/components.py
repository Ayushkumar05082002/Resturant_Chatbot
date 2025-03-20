import streamlit as st
from datetime import datetime, timedelta

def render_navigation():
    """Render the navigation bar"""
    st.markdown("""
    <div class="nav-bar">
        <a href="#home">Home</a>
        <a href="#about-us">About Us</a>
        <a href="#menu-query">Menu Query</a>
        <a href="#table-reservation">Table Reservation</a>
        <a href="#contact-us">Contact Us</a>
    </div>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main header"""
    st.markdown('<div class="main-header">🍽️ Welcome To STOA</div>', unsafe_allow_html=True)

def get_available_times():
    """Get list of available reservation times"""
    return ["5:00 PM", "5:30 PM", "6:00 PM", "6:30 PM", "7:00 PM", 
            "7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM"]
