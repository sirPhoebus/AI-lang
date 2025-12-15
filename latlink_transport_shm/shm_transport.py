import time
import struct
import logging
from multiprocessing import shared_memory
from typing import Optional

# Layout of the control block (first 128 bytes reserved)
# [0:8]   Write Cursor (int64) - monotonic
# [8:16]  Read Cursor (int64)  - monotonic
# [16:24] Buffer Capacity (int64)
# [24:32] Magic/Version
CONTROL_BLOCK_SIZE = 128
CONTROL_MAGIC_VERSION = 0x4C41544C494E4B01

class ShmRingBuffer:
    def __init__(self, name: str, size: int, create: bool = False):
        self.logger = logging.getLogger(f"ShmBuffer-{name}")

        self.shm = None
        self.buf = None

        try:
            if create:
                # Cleanup previous if exists? 
                # Ideally, we unlink if it exists, but that might break other process.
                # simpler to just create.
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                self.buf = self.shm.buf
            else:
                self.shm = shared_memory.SharedMemory(name=name, create=False)
                self.buf = self.shm.buf
        except FileExistsError:
            # If create=True but it exists, just open it
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            self.buf = self.shm.buf

        if self.buf is None:
            raise RuntimeError("Shared memory buffer not initialized")

        self.size = int(self.shm.size)
        self.capacity = self.size - CONTROL_BLOCK_SIZE
        if self.capacity <= 0:
            raise ValueError("Size must be larger than control block")

        if create:
            self._write_control(0, 0)
            struct.pack_into("!Q", self.buf, 16, self.capacity)
            struct.pack_into("!Q", self.buf, 24, CONTROL_MAGIC_VERSION)
        else:
            stored_capacity = struct.unpack_from("!Q", self.buf, 16)[0]
            stored_magic = struct.unpack_from("!Q", self.buf, 24)[0]
            if stored_magic != CONTROL_MAGIC_VERSION:
                raise RuntimeError("Invalid SHM control block magic/version")
            if int(stored_capacity) != int(self.capacity):
                raise RuntimeError("SHM capacity mismatch")

    def _write_control(self, w_cursor: int, r_cursor: int):
        struct.pack_into("!QQ", self.buf, 0, w_cursor, r_cursor)

    def _read_cursors(self):
        # returns (write_cursor, read_cursor)
        return struct.unpack_from("!QQ", self.buf, 0)

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()

    def write_bytes(self, data: bytes, timeout: float = 5.0) -> int:
        start_time = time.time()
        data_len = len(data)
        if data_len > self.capacity:
            raise ValueError("Write exceeds ring buffer capacity")

        while True:
            w, r = self._read_cursors()
            used = w - r
            free = self.capacity - used

            if free >= data_len:
                # Write data
                # Calculate wrap-around
                start_offset = CONTROL_BLOCK_SIZE + (w % self.capacity)

                # Check for wrap
                first_chunk_len = min(data_len, self.capacity - (w % self.capacity))
                self.buf[start_offset : start_offset + first_chunk_len] = data[:first_chunk_len]
                
                if first_chunk_len < data_len:
                    # Write wrapped part
                    remaining = data_len - first_chunk_len
                    self.buf[CONTROL_BLOCK_SIZE : CONTROL_BLOCK_SIZE + remaining] = data[first_chunk_len:]
                
                # Commit: update write cursor
                # ensure data is visible before cursor update (memory barrier concept)
                # In Python, we can't easily force this, but order of ops usually holds on x86
                struct.pack_into("!Q", self.buf, 0, w + data_len)
                return data_len
            
            if time.time() - start_time > timeout:
                raise TimeoutError("Buffer full, timeout writing")
            
            time.sleep(0.001) # Spin wait 1ms

    def read_bytes(self, num_bytes: int, timeout: float = 5.0) -> bytes:
        start_time = time.time()
        
        while True:
            w, r = self._read_cursors()
            available = w - r
            
            if available >= num_bytes:
                # Read data
                start_offset = CONTROL_BLOCK_SIZE + (r % self.capacity)
                
                first_chunk_len = min(num_bytes, self.capacity - (r % self.capacity))
                chunk1 = bytes(self.buf[start_offset : start_offset + first_chunk_len])
                
                chunk2 = b''
                if first_chunk_len < num_bytes:
                    remaining = num_bytes - first_chunk_len
                    chunk2 = bytes(self.buf[CONTROL_BLOCK_SIZE : CONTROL_BLOCK_SIZE + remaining])
                
                data = chunk1 + chunk2
                
                # Commit: update read cursor
                struct.pack_into("!Q", self.buf, 8, r + num_bytes)
                return data
                
            if time.time() - start_time > timeout:
                return None # Or raise TimeoutError
            
            time.sleep(0.001)

    def readable_bytes(self) -> int:
        w, r = self._read_cursors()
        return w - r
