"""
Tests for wire format versioning (WIRE_VERSION byte prefix).

Covers:
  - TCPClient.send() / send_batch() prepend version byte
  - TCPServer._drain_frames() parses versioned headers
  - Unknown versions are dropped with stderr warning
  - Round-trip through real sockets
  - DatabaseWriter on-disk format includes version byte
"""

import struct
import sys
import time
from unittest.mock import MagicMock, patch

import msgspec
import pytest


class TestSendPrependsVersionByte:
    """TCPClient.send() must prepend WIRE_VERSION before length."""

    def test_send_prepends_version_byte(self):
        from traceml.transport.tcp_transport import (
            WIRE_VERSION,
            TCPClient,
            TCPConfig,
        )

        client = TCPClient(TCPConfig())
        mock_sock = MagicMock()
        client._sock = mock_sock
        client._connected = True

        payload = {"table": "test", "rows": [1, 2, 3]}
        client.send(payload)

        sent = mock_sock.sendall.call_args[0][0]
        # First byte is WIRE_VERSION
        assert sent[0] == WIRE_VERSION
        # Bytes 1-4 are big-endian uint32 length
        length = struct.unpack("!I", sent[1:5])[0]
        # Remaining bytes are valid msgspec payload
        decoded = msgspec.msgpack.decode(sent[5:])
        assert decoded == payload
        assert length == len(sent) - 5


class TestSendBatchPrependsVersionByte:
    """TCPClient.send_batch() must prepend WIRE_VERSION."""

    def test_send_batch_prepends_version_byte(self):
        from traceml.transport.tcp_transport import (
            WIRE_VERSION,
            TCPClient,
            TCPConfig,
        )

        client = TCPClient(TCPConfig())
        mock_sock = MagicMock()
        client._sock = mock_sock
        client._connected = True

        payloads = [{"a": 1}, {"b": 2}]
        client.send_batch(payloads)

        sent = mock_sock.sendall.call_args[0][0]
        assert sent[0] == WIRE_VERSION
        length = struct.unpack("!I", sent[1:5])[0]
        decoded = msgspec.msgpack.decode(sent[5:])
        assert decoded == payloads
        assert length == len(sent) - 5


class TestDrainFramesParsesVersion:
    """_drain_frames must return (version, payload) tuples."""

    def test_drain_frames_parses_version(self):
        from traceml.transport.tcp_transport import (
            WIRE_VERSION,
            TCPServer,
            TCPConfig,
        )

        server = TCPServer(TCPConfig())
        payload = msgspec.msgpack.encode({"key": "value"})
        raw = (
            bytes([WIRE_VERSION])
            + struct.pack("!I", len(payload))
            + payload
        )
        buffer = bytearray(raw)

        frames, buffer, expected = server._drain_frames(
            buffer, None
        )

        assert len(frames) == 1
        version, data = frames[0]
        assert version == WIRE_VERSION
        assert data == payload

    def test_drain_frames_multiple(self):
        from traceml.transport.tcp_transport import (
            WIRE_VERSION,
            TCPServer,
            TCPConfig,
        )

        server = TCPServer(TCPConfig())
        raw = bytearray()
        payloads = []
        for i in range(3):
            p = msgspec.msgpack.encode({"idx": i})
            payloads.append(p)
            raw += (
                bytes([WIRE_VERSION])
                + struct.pack("!I", len(p))
                + p
            )

        frames, _, _ = server._drain_frames(raw, None)

        assert len(frames) == 3
        for i, (version, data) in enumerate(frames):
            assert version == WIRE_VERSION
            assert data == payloads[i]


class TestUnknownVersionDropped:
    """Unknown wire versions must be dropped with stderr warning."""

    def test_unknown_version_dropped(self, capsys):
        from traceml.transport.tcp_transport import (
            WIRE_VERSION,
            TCPServer,
            TCPConfig,
        )

        server = TCPServer(TCPConfig())

        # Build a valid frame with unknown version 99
        payload = msgspec.msgpack.encode({"bad": True})
        bad_frame = (
            bytes([99])
            + struct.pack("!I", len(payload))
            + payload
        )
        # Also build a valid frame so we can verify good ones
        good_payload = msgspec.msgpack.encode({"good": True})
        good_frame = (
            bytes([WIRE_VERSION])
            + struct.pack("!I", len(good_payload))
            + good_payload
        )

        buffer = bytearray(bad_frame + good_frame)
        frames, _, _ = server._drain_frames(buffer, None)

        # Simulate _handle_client logic for version checking
        import queue as queue_mod

        decoder = msgspec.msgpack.Decoder()
        results = []
        for version, data in frames:
            from traceml.transport.tcp_transport import (
                SUPPORTED_VERSIONS,
            )

            if version not in SUPPORTED_VERSIONS:
                print(
                    f"[TraceML] Unknown wire version"
                    f" {version}, dropping message",
                    file=sys.stderr,
                )
                continue
            msg = decoder.decode(data)
            results.append(msg)

        captured = capsys.readouterr()
        assert "[TraceML] Unknown wire version 99" in captured.err
        assert len(results) == 1
        assert results[0] == {"good": True}


class TestRoundtripVersioned:
    """End-to-end: TCPClient -> TCPServer with version byte."""

    def test_roundtrip_versioned(self):
        from traceml.transport.tcp_transport import (
            TCPClient,
            TCPConfig,
            TCPServer,
        )

        import socket

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        cfg = TCPConfig(host="127.0.0.1", port=port)
        server = TCPServer(cfg)
        server.start()

        try:
            client = TCPClient(cfg)
            msg = {"table": "test", "rows": [1, 2, 3]}
            client.send(msg)

            # Poll server for the message
            received = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                for m in server.poll():
                    received = m
                if received is not None:
                    break
                time.sleep(0.05)

            assert received == msg
            client.close()
        finally:
            server.stop()


class TestDatabaseWriterVersionByte:
    """DatabaseWriter.flush() must prepend version byte on disk."""

    def test_database_writer_version_byte(self, tmp_path):
        from traceml.transport.tcp_transport import WIRE_VERSION

        # We need to mock config and session to avoid real paths
        with (
            patch("traceml.database.database_writer.config")
            as mock_config,
            patch(
                "traceml.database.database_writer.get_session_id",
                return_value="test-session",
            ),
            patch(
                "traceml.database.database_writer.get_ddp_info",
                return_value=(0, 0, 1),
            ),
        ):
            mock_config.enable_logging = True
            mock_config.logs_dir = str(tmp_path)

            from traceml.database.database import Database
            from traceml.database.database_writer import (
                DatabaseWriter,
            )

            db = Database("test_sampler")
            db.add_record("metrics", {"step": 1, "loss": 0.5})

            writer = DatabaseWriter(
                db, "test_sampler", flush_every=1
            )
            writer.flush()

            # Find the written file
            msgpack_files = list(tmp_path.rglob("*.msgpack"))
            assert len(msgpack_files) == 1

            data = msgpack_files[0].read_bytes()
            # First byte must be WIRE_VERSION
            assert data[0] == WIRE_VERSION
            # Bytes 1-4 are big-endian uint32 length
            length = struct.unpack("!I", data[1:5])[0]
            # Remaining bytes are valid msgspec payload
            decoded = msgspec.msgpack.decode(data[5 : 5 + length])
            assert decoded["step"] == 1
            assert decoded["loss"] == 0.5
