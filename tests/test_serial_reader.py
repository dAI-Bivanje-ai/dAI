from unittest.mock import Mock
from src.realtime.serial_reader import LiveSerialReader
from unittest.mock import patch


def test_read_chunk_returns_bytes():
    """
    Test preveri realtime branje bajtov iz serial povezave.
    -------------------------
    LiveSerialReader v pravem programu ne bere datoteke,
    ampak neprekinjeno bere nove bajte direktno iz STM32
    preko USB serial povezave.

    V testu simuliramo realtime STM32 tako, da ustvarimo
    fake serial objekt (Mock), ki se obnaša kot prava
    serial povezava.

    Namesto dejanskega STM32:
        STM32 -> serial.read()

    uporabimo:
        fake_serial -> fake_serial.read()

    Test preveri:
    - ali read_chunk() pravilno kliče serial.read()
    - ali vrne pravilne realtime bajte
    - ali reader pravilno uporablja serial API
    """
    #ustvari fake serial povezavo
    fake_serial = Mock()
    # simulirani realtime podatki iz STM32
    fake_serial.read.return_value = b"\xff\xff\x01abc"
    # simuliranje odprte povezave
    fake_serial.is_open = True

    # ustvari realtime reader
    reader = LiveSerialReader()
    # namesto prave povezave = fake povezava
    reader.connection = fake_serial
    # realtime reader prebere 6 bajtov
    data = reader.read_chunk(6)
    # preverimo ce smo dobili prave bajte
    assert data == b"\xff\xff\x01abc"
    # ali je bil serial.read() klican pravilno
    fake_serial.read.assert_called_once_with(6)

def test_read_chunk_without_connection_raises():
    """
    Test preveri varnost realtime readerja.

    Realtime sistem nikoli ne sme brati podatkov,
    če STM32 ni povezan.

    Test preveri:
    - ali read_chunk() pravilno sproži RuntimeError
    - ali program ne poskuša brati iz neobstoječe povezave
    """
    reader = LiveSerialReader()

    try:
        # poskus realtime branja brez povezave
        reader.read_chunk()
        # če pridemo sem, test ni pravilen
        assert False

    except RuntimeError:
        # pricakovan rezultat
        assert True

@patch("serial.tools.list_ports.comports")
def test_find_stm32_port(mock_comports):
    """
    Test preveri realtime zaznavanje STM32 naprave.

    LiveSerialReader mora v realtime sistemu samodejno:
    - poiskati STM32
    - zaznati pravi USB port
    - vzpostaviti povezavo

    V testu simuliramo seznam USB naprav,
    ki bi jih Linux vrnil preko:
        serial.tools.list_ports.comports()

    Test preveri:
    - ali reader pravilno prepozna STM32 VID/PID
    - ali vrne pravilen serial port

    To omogoča:
    - auto reconnect
    - hot plug STM32
    - stabilen realtime sistem
    """

    fake_port = Mock()
    # simuliramo STM32 VID/PID
    fake_port.hwid = "USB VID:PID=0483:5740"
    # simuliramo Linux serial port
    fake_port.device = "/dev/ttyACM0"
    # mock funkcija vrne naš fake port
    mock_comports.return_value = [fake_port]

    reader = LiveSerialReader()
    # reader poišče STM32
    port = reader.find_stm32_port()
    # preverimo ali je našel pravilen port
    assert port == "/dev/ttyACM0"