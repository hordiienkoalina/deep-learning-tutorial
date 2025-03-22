import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0)) # Bind to any available port
print(f"Available port: {s.getsockname()[1]}")
s.close()