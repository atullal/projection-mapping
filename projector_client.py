"""Tiny client for projector.py (stdlib only, safe to import from the cv2 side)."""
import socket
import sys

PORT = 47800


def send(command, timeout=20):
    with socket.create_connection(("127.0.0.1", PORT), timeout=timeout) as conn:
        conn.sendall((command + "\n").encode())
        reply = conn.makefile().readline().strip()
    if not reply.startswith("ok"):
        raise RuntimeError(f"projector: {reply!r} for {command!r}")
    return reply


if __name__ == "__main__":
    print(send(" ".join(sys.argv[1:])))
