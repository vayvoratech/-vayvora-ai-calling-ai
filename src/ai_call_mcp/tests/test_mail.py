import os
import smtplib
import imaplib
import email
import time
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gmail Server Configurations
MAIL_HOST_SMTP = os.getenv("MAIL_HOST_SMTP", "smtp.gmail.com")
MAIL_HOST_IMAP = os.getenv("MAIL_HOST_IMAP", "imap.gmail.com")
MAIL_PORT_SMTP = int(os.getenv("MAIL_PORT_SMTP", "587"))
MAIL_PORT_IMAP = int(os.getenv("MAIL_PORT_IMAP", "993"))
MAIL_USER = os.getenv("MAIL_USER", "")
MAIL_PASS = os.getenv("MAIL_PASS", "")

# --- CHANGE THIS TO WHO YOU WANT TO SEND THE EMAIL TO ---
TO_EMAIL = MAIL_USER  # Defaults to sending to yourself for testing


def test_smtp_send():
    print("--- 1. Testing SMTP (Sending Email) ---")
    if not MAIL_USER or not MAIL_PASS:
        print("Error: MAIL_USER or MAIL_PASS not set in your .env file.")
        return False

    msg = EmailMessage()
    msg.set_content(
        "Hello! This is a test email sent natively from your Python script using Gmail SMTP."
    )
    msg["Subject"] = "MCP Mail Test Success"
    msg["From"] = MAIL_USER
    msg["To"] = "pawanganesh21511111@gmail.com"

    try:
        print(f"Connecting to SMTP server {MAIL_HOST_SMTP}:{MAIL_PORT_SMTP}...")
        with smtplib.SMTP(MAIL_HOST_SMTP, MAIL_PORT_SMTP) as server:
            server.starttls()  # Secure the connection
            server.login(MAIL_USER, MAIL_PASS)
            server.send_message(msg)
        print(f"Success! Email successfully sent to {TO_EMAIL}.")
        return True
    except Exception as e:
        print(f"SMTP Send Failed: {e}")
        return False


def test_imap_read():
    print("\n--- 2. Testing IMAP (Reading Inbox) ---")
    try:
        print(f"Connecting to IMAP server {MAIL_HOST_IMAP}:{MAIL_PORT_IMAP}...")
        mail = imaplib.IMAP4_SSL(MAIL_HOST_IMAP, MAIL_PORT_IMAP)
        mail.login(MAIL_USER, MAIL_PASS)
        mail.select("inbox")

        status, messages = mail.search(None, "ALL")
        if status != "OK":
            print("Could not retrieve inbox messages.")
            return

        mail_ids = messages[0].split()
        print(f"Success! Connected to inbox. Total messages: {len(mail_ids)}")

        if mail_ids:
            # Fetch the most recent message
            latest_id = mail_ids[-1]
            status, data = mail.fetch(latest_id, "(RFC822)")
            if status == "OK":
                msg = email.message_from_bytes(data[0][1])
                print(
                    f"Latest Message -> From: {msg['From']} | Subject: {msg['Subject']}"
                )

        mail.logout()
    except Exception as e:
        print(f"IMAP Read Failed: {e}")


if __name__ == "__main__":
    # Step 1: Send the email
    email_sent = test_smtp_send()

    # Step 2: If sent successfully, wait briefly and check the inbox via IMAP
    if email_sent:
        print("\nWaiting 3 seconds for email to sync in inbox...")
        time.sleep(3)
        test_imap_read()