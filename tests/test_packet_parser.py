import struct

from src.realtime.packet_parser import LivePacketParser, SYNC


def make_packet(counter: int, timestamp: int, chunks_data: bytes) -> bytes:
    """
    Helper za izdelavo veljavnega testnega paketa.

    Paket naredimo umetno, zato ne rabimo STM32.
    """
    parser = LivePacketParser()

    payload_without_crc = (
        struct.pack("<I", timestamp)
        + struct.pack("<H", 6 + len(chunks_data) + 2 - 1)
        + chunks_data
    )

    crc = parser.data_logger.crc16_compute(payload_without_crc)
    payload = payload_without_crc + struct.pack("<H", crc)

    return SYNC + bytes([counter]) + payload


def make_imu_chunk(chunk_id: int, samples: list[tuple[int, int, int]]) -> bytes:
    """
    Naredi gyro/acc chunk.
    """
    data = b"".join(struct.pack("<hhh", x, y, z) for x, y, z in samples)
    size = len(data)

    return (
        bytes([chunk_id])
        + struct.pack("<H", size - 1)
        + b"\x00"
        + data
    )


def make_mic_chunk(samples: list[int]) -> bytes:
    """
    Naredi mic chunk.
    """
    data = bytes((sample & 0xFF for sample in samples))
    size = len(data)

    return (
        b"\x04"
        + struct.pack("<H", size - 1)
        + b"\x00"
        + data
    )


def test_feed_keeps_incomplete_packet():
    parser = LivePacketParser()

    packets = parser.feed(b"\xff\xff\x01abc")

    assert packets == []
    assert parser.buffer.startswith(SYNC)


def test_feed_removes_garbage_before_sync():
    parser = LivePacketParser()

    parser.feed(b"garbage" + SYNC + b"\x01abc")

    assert parser.buffer.startswith(SYNC)


def test_parse_valid_acc_packet():
    parser = LivePacketParser()

    acc_chunk = make_imu_chunk(
        2,
        [(1, 2, 3), (4, 5, 6)],
    )

    packet_bytes = make_packet(
        counter=1,
        timestamp=1234,
        chunks_data=acc_chunk,
    )

    # Dodamo še drugi SYNC, da feed() ve, da je prvi paket zaključen.
    packets = parser.feed(packet_bytes + SYNC)

    assert len(packets) == 1
    assert packets[0]["packet_counter"] == 1
    assert packets[0]["timestamp"] == 1234
    assert packets[0]["chunks"][2] == [(1, 2, 3), (4, 5, 6)]


def test_parse_valid_gyro_and_acc_packet():
    parser = LivePacketParser()

    gyro_chunk = make_imu_chunk(1, [(10, 20, 30)])
    acc_chunk = make_imu_chunk(2, [(1, 2, 3)])

    packet_bytes = make_packet(
        counter=7,
        timestamp=5000,
        chunks_data=gyro_chunk + acc_chunk,
    )

    packets = parser.feed(packet_bytes + SYNC)

    assert len(packets) == 1
    assert packets[0]["chunks"][1] == [(10, 20, 30)]
    assert packets[0]["chunks"][2] == [(1, 2, 3)]


def test_parse_valid_mic_packet():
    parser = LivePacketParser()

    mic_chunk = make_mic_chunk([1, 2, 255])

    packet_bytes = make_packet(
        counter=3,
        timestamp=999,
        chunks_data=mic_chunk,
    )

    packets = parser.feed(packet_bytes + SYNC)

    assert len(packets) == 1

    # int8: 255 pomeni -1
    assert packets[0]["chunks"][4] == [1, 2, -1]


def test_bad_crc_packet_is_rejected():
    parser = LivePacketParser()

    acc_chunk = make_imu_chunk(2, [(1, 2, 3)])
    packet_bytes = bytearray(make_packet(1, 100, acc_chunk))

    # Pokvarimo en bajt v paketu.
    packet_bytes[-1] ^= 0xFF

    packets = parser.feed(bytes(packet_bytes) + SYNC)

    assert packets == []
    assert parser.invalid_packets == 1