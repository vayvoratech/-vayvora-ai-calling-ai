import asyncio
import os
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Ensure .env is loaded before importing and registering tools
load_dotenv()

from src.ai_call_mcp.tools.mail import register_mail_tools


async def run_tests():
    print("=" * 60)
    print("🧪 Running FastMCP Mail Tools Test Suite")
    print("=" * 60)

    # 1. Read environment settings
    smtp_host = os.getenv("MAIL_HOST_SMTP", "localhost")
    smtp_port = os.getenv("MAIL_PORT_SMTP", "587")
    imap_host = os.getenv("MAIL_HOST_IMAP", "localhost")
    imap_port = os.getenv("MAIL_PORT_IMAP", "993")
    mail_user = os.getenv("MAIL_USER", "")
    mail_pass = os.getenv("MAIL_PASS", "")

    print("\n⚙️  Loaded Environment Config:")
    print(f" • SMTP Server : {smtp_host}:{smtp_port}")
    print(f" • IMAP Server : {imap_host}:{imap_port}")
    print(f" • Mail User   : {mail_user or '[Not Set / Localhost Fallback]'}")
    print(f" • Mail Pass   : {'********' if mail_pass else '[Not Set]'}")

    # 2. Initialize FastMCP and register tools with loaded environment
    mcp = FastMCP("VayvoraMailServer")
    register_mail_tools(mcp)

    registered_tools = await mcp.list_tools()
    tool_names = [t.name for t in registered_tools]
    print(f"\n📋 Registered Tools: {tool_names}")
    assert "mail_send" in tool_names, "mail_send was not registered!"
    assert "mail_read_recent" in tool_names, "mail_read_recent was not registered!"

    target_email = mail_user or "kirankanteti143@gmail.com"

    # -------------------------------------------------------------
    # Test 1: Input Validation Guards
    # -------------------------------------------------------------
    print("\n--- Test 1: Input Validation Guards ---")
    
    # Must pass an empty string to verify the guard triggers
    try:
        await mcp.call_tool(
            name="mail_send",
            arguments={"to": "   ", "subject": "Hello", "body": "Body"},
        )
        print("❌ Failed: Empty 'to' did not raise an error.")
    except Exception as e:
        print(f"✅ Caught expected error for empty 'to': {e}")

    try:
        await mcp.call_tool(
            name="mail_send",
            arguments={"to": target_email, "subject": "   ", "body": "Body"},
        )
        print("❌ Failed: Empty 'subject' did not raise an error.")
    except Exception as e:
        print(f"✅ Caught expected error for empty 'subject': {e}")

    try:
        await mcp.call_tool(
            name="mail_read_recent",
            arguments={"limit": 0},
        )
        print("❌ Failed: Invalid 'limit' did not raise an error.")
    except Exception as e:
        print(f"✅ Caught expected error for limit < 1: {e}")

    # -------------------------------------------------------------
    # Test 2: mail_send (Using .env Configuration with Mocking)
    # -------------------------------------------------------------
    print("\n--- Test 2: mail_send Execution Flow (Mocked Network) ---")
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        res_send = await mcp.call_tool(
            name="mail_send",
            arguments={
                "to": target_email,
                "subject": "System Verification",
                "body": "Automated verification test.",
            },
        )
        print(f"Result: {res_send}")

        if mail_user and mail_pass and smtp_host != "localhost":
            print(f"SMTP Connected to : {mock_smtp.call_args[0][0]}:{mock_smtp.call_args[0][1]}")
            print(f"StartTLS Called   : {mock_server.starttls.called}")
            print(f"Login Called      : {mock_server.login.called}")
            print(f"Send Called       : {mock_server.send_message.called}")
        else:
            print("ℹ️ Fallback/localhost mode used (No remote SMTP invoked).")

    # -------------------------------------------------------------
    # Test 3: mail_read_recent (Using .env Configuration with Mocking)
    # -------------------------------------------------------------
    print("\n--- Test 3: mail_read_recent Execution Flow (Mocked Network) ---")
    with patch("imaplib.IMAP4_SSL") as mock_imap:
        mock_client = MagicMock()
        mock_imap.return_value = mock_client

        mock_client.select.return_value = ("OK", [b"2"])
        mock_client.search.return_value = ("OK", [b"1 2"])

        raw_email = (
            f"From: {target_email}\r\n"
            "Subject: Test Subject\r\n\r\n"
            "Test body content"
        ).encode("utf-8")
        mock_client.fetch.return_value = ("OK", [(b"2 (RFC822)", raw_email)])

        res_read = await mcp.call_tool(
            name="mail_read_recent",
            arguments={"limit": 2},
        )
        print(f"Fetched Output:\n{res_read}")
        print(f"IMAP Logout Called: {mock_client.logout.called}")

    # -------------------------------------------------------------
    # Test 4: Real SMTP Live Check (Optional / Guarded)
    # -------------------------------------------------------------
    if mail_user and mail_pass and smtp_host != "localhost":
        print("\n--- Test 4: Live Network Test (Sending real email) ---")
        try:
            live_res = await mcp.call_tool(
                name="mail_send",
                arguments={
                    "to": target_email,
                    "subject": "Vayvora Voice AI - Live Connection Test",
                    "body": "This is a real live test verifying the .env mail credentials.",
                },
            )
            print(f"Live Send Result: {live_res}")
        except Exception as live_err:
            print(f"Live Test Failed: {live_err}")

    print("\n" + "=" * 60)
    print("🎉 Test suite completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())