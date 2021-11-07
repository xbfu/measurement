import socket
import time


class SocketMsger:
    def __init__(self, socket, is_listener=False):
        self.closed = False
        self.rv_buffer = ""
        self.socket = socket
        self.is_listener = is_listener
        self.data = None

    def recv(self):
        if self.is_listener:
            return
        if self.closed:
            return None
        self.rv_buffer = self.rv_buffer.lstrip("\0")
        i = self.rv_buffer.find("\0")
        if i > 0:
            rt = self.rv_buffer[:i]
            self.rv_buffer = self.rv_buffer[i + 1 :]
            return rt
        else:
            while i < 0:
                new_string = self.socket.recv(1024).decode()
                if new_string == "":
                    self.closed = True
                    return None
                self.rv_buffer = (self.rv_buffer + new_string).lstrip("\0")
                i = self.rv_buffer.find("\0")
            rt = self.rv_buffer[:i]
            self.rv_buffer = self.rv_buffer[i + 1 :]
            return rt

    def send(self, string):
        if self.is_listener:
            return
        if self.closed:
            return None
        s = string + "\0"
        self.socket.sendall(s.encode())

    def close(self):
        self.socket.close()
        self.closed = True

    @staticmethod
    def tcp_listener(listen_ip, listen_port, backlog=100):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listen_ip, listen_port))
        listener.listen(backlog)
        return SocketMsger(listener, True)

    def accept(self):
        if self.is_listener:
            conn, address = self.socket.accept()
            connm = SocketMsger(conn)
            return (connm, address)

    @staticmethod
    def tcp_connect(ip, port, retry=True):
        sock = None
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((ip, port))
                return SocketMsger(sock)
            except Exception as e:
                print("Server not found or not open!", e)
                if not retry:
                    return None
                time.sleep(3)
