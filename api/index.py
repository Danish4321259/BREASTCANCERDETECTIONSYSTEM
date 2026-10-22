"""
Vercel Serverless Function entry point
Routes API requests to the Flask application.
"""
from app import app

# Vercel WSGI handler
handler = app
