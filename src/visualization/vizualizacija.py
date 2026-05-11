import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from data_logger.data_logger import parse_file


ID_GYRO = 1
ID_ACC = 2
ID_MAG = 3

BYTES_PER_VALUE = {
    ID_GYRO: 2,
    ID_ACC: 2,
    ID_MAG: 2,
}

RES = {ID_GYRO: 8.75e-3, ID_ACC: 1e-3, ID_MAG: 1.5e-3}

UNITS = {ID_GYRO: "°/s", ID_ACC: "g", ID_MAG: "mGauss"}

SENSOR_NAME = {ID_GYRO: "Gyroscope", ID_ACC: "Accelerometer", ID_MAG: "Magnetometer"}


class Packet:
    """
    Razred za predstavitev enega podatkovnega paketa iz senzorja.

    Atributi:
    id (int): Identifikator senzorja (1 = giroskop, 2 = pospeskometer, 3 = magnetometer).
    ts (float): casovna oznaka paketa v milisekundah od zacetka snemanja.
    data (np.ndarray): Surovi bajti vzorcev, shranjeni kot uint8 NumPy polje.
    """

    def __init__(self, id: int, ts: float, data: np.ndarray):
        self.id = id
        self.ts = ts
        self.data = data


def parsed_packets_to_class(packets: List) -> List:
    """
    Pretvori seznam slovarjev (izhod iz parse_file) v seznam objektov razreda Packet.

    Vsak vhodni slovar vsebuje timestamp in slovar blokov ("chunks"), kjer je
    kljuc ID senzorja, vrednost pa seznam trojk (x, y, z) surovih vzorcev.
    Vzorci se pretvorijo v int16 bajte in shranijo v objekt Packet.

    Parameters:
    packets (List[dict]): Izhod funkcije parse_file — seznam razclenjenih paketov.

    Returns:
    List[Packet]: Seznam objektov Packet, ki vsebujejo surove bajte in metapodatke.

    Opomba:
    Bloki z ID-ji, ki niso definirani v BYTES_PER_VALUE, se preskocijo.

    """
    new_packets = []

    for p in packets:
        ts = p["timestamp"]
        for chunk_id, samples in p["chunks"].items():
            if chunk_id not in BYTES_PER_VALUE:
                continue

            raw = bytearray()
            for x, y, z in samples:
                raw += np.int16(x).tobytes()
                raw += np.int16(y).tobytes()
                raw += np.int16(z).tobytes()

            packet = Packet(
                id=chunk_id,
                ts=float(ts),
                data=np.frombuffer(bytes(raw), dtype=np.uint8),
            )
            new_packets.append(packet)
    return new_packets


def sestavi_podatke(packet_list: list) -> Tuple[float, np.ndarray]:
    """
    Iz seznama paketov enega senzorja izracuna vzorcevalno frekvenco in sestavi
    matriko vzorcev z vsemi tremi osmi (X, Y, Z), pretvorjeno v fizikalne enote.

    Vzorcevalna frekvenca se oceni kot razmerje med povprecni stevilom vzorcev na paket (avg Nvz)
    in povprecnim casovnim razmikom med paketi (avg (Tn+1 - Tn)):
        Fvz = avg(Nvz)/avg(Tn+1 - Tn)
    Surove vrednosti int16 se pomnozijo z resolucijo senzorja (RES), da dobimo fizikalne enote (°/s, g, mGauss).

    Parameters:
    packet_list (List[Packet]): Seznam paketov istega senzorja, urejen po casu.

    Returns:
    Tuple[float, np.ndarray]:
        - Fvz (float): Ocenjena vzorcevalna frekvenca v Hz.
        - signal (np.ndarray): Matrika oblike (N, 3) s pretvorjenimi vzorci.

    Exceptions:
    ValueError: ce je seznam prazen, vsebuje premalo paketov ali nobenih veljavnih vzorcev
    """
    if not packet_list:
        raise ValueError("Seznam paketov je prazen")

    sensor_id = packet_list[0].id
    bytes_per_val = BYTES_PER_VALUE.get(sensor_id, 2)
    bytes_per_sample = bytes_per_val * 3  # 3 osi: x,y,z

    T_packets = []
    Nvz_packets = []

    # izracun casovnih razmikov med paketi in stevilo vzorcev na paket
    for i in range(1, len(packet_list)):
        dt = packet_list[i].ts - packet_list[i - 1].ts
        if dt > 0:
            T_packets.append(dt / 1000.0)  # pretvorba iz ms v s

        n_bytes = len(packet_list[i].data)
        n_samples = n_bytes // bytes_per_sample

        if n_samples > 0:
            Nvz_packets.append(n_samples)

    if not T_packets or not Nvz_packets:
        raise ValueError("Premalo paketov za izracun Fvz")

    T_avg = np.mean(T_packets)
    Nvz_avg = np.mean(Nvz_packets)

    # ocena vzorcevalne frekvence
    Fvz = Nvz_avg / T_avg

    samples = []

    for p in packet_list:
        n_bytes = len(p.data)
        n_samples = n_bytes // bytes_per_sample

        for i in range(n_samples):
            start = i * bytes_per_sample
            chunk = p.data[start : start + bytes_per_sample]
            # dekodiranje surovih bajtov v int16 trojice (little-endian)
            values = np.frombuffer(chunk.tobytes(), dtype="<i2")
            if len(values) == 3:
                samples.append(values)

    if not samples:
        raise ValueError("Ni bil omogoce sestaviti vzorcev")

    raw_signal = np.array(samples, dtype=float)

    resolution = RES.get(sensor_id, 1.0)
    # pretvorba surovih vrednosti v fizikalne enote z mnozenjem z resolucijo senzorja
    signal = raw_signal * resolution

    return float(Fvz), signal


def prikazi_signal(
    signal: np.ndarray,
    title: Optional[str] = None,
    start_i: Optional[int] = None,
    end_i: Optional[int] = None,
    unit: str = "",
    Fvz: Optional[float] = None,
) -> None:
    """
    Izris signala senzorja za vse tri osi (X, Y, Z) v odvisnosti od casa ali
    zaporedne stevilke vzorca.

    Ce je podana vzorcevalna frekvenca (Fvz), se na x-osi prikaze cas v sekundah;
    sicer se prikaze indeks vzorca. Opcijsko je mogoce izrisati le dolocen odsek
    signala s parametroma start_i in end_i.

    Parameters:
    signal (np.ndarray): Matrika vzorcev oblike (N, 3) ali vektor dolzine N.
    title (str, optional): Naslov grafa. Privzeto brez naslova.
    start_i (int, optional): Zacetni indeks izreza signala (vkljucno).
    end_i (int, optional): Koncni indeks izreza signala (izkljucno).
    unit (str): Enota na osi Y (npr. "°/s", "g", "mGauss", "Amplituda").
    Fvz (float, optional): Vzorcevalna frekvenca v Hz za pretvorbo indeksov v cas.
    """

    if start_i is not None and end_i is not None:
        cut = signal[start_i:end_i]
        idx_start = start_i
    elif start_i is not None:
        cut = signal[start_i:]
        idx_start = start_i
    elif end_i is not None:
        cut = signal[:end_i]
        idx_start = 0
    else:
        cut = signal
        idx_start = 0

    N = len(cut)

    if Fvz and Fvz > 0:
        t = (np.arange(N) + idx_start) / Fvz
        x_label = "Cas [s]"

    else:
        t = np.arange(N) + idx_start
        x_label = "Vzorec [#]"

    fig, ax = plt.subplots(figsize=(12, 5), num=f"Graf - {title}")

    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["X", "Y", "Z"]

    if cut.ndim == 2 and cut.shape[1] >= 3:
        for k in range(3):
            ax.plot(t, cut[:, k], color=colors[k], label=labels[k], linewidth=0.8)
    else:
        ax.plot(t, cut, color="tab:blue", linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(unit if unit else "Amplituda")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    #    Vstopna tocka programa. Izvede naslednje korake:

    #    1. Prebere in razcleni binarno datoteko "filename".bin s funkcijo parse_file.
    #    2. Pakete loci po tipu senzorja (giroskop, pospeskometer, magnetometer).
    #    3. Za vsak senzor sestavi signal in izracuna vzorcevalno frekvenco (sestavi_podatke).
    #    4. Izpise stevilo vzorcev in frekvenco za vsak senzor.
    #    5. Izrise celoten signal za vsak senzor.
    #    6. Izrise odsek giroskopskega signala med 2. in 5. sekundo.

    filename = "LOG03.bin"

    parsed_packets = parse_file(filename)

    def filter_by_id(sensor_id):
        """
        Pomožna funkcija za filtriranje paketov po ID-ju senzorja.

        Parameters:
        sensor_id (int): ID senzorja, po katerem filtriramo.

        Returns:
        List[Packet]: Paketi z ustreznim sensor_id.

        """
        packets = parsed_packets_to_class(parsed_packets)
        return [p for p in packets if p.id == sensor_id]

    gyro_packets = filter_by_id(ID_GYRO)
    acc_packets = filter_by_id(ID_ACC)
    mag_packets = filter_by_id(ID_MAG)

    print(
        f"Gyro: od {gyro_packets[0].ts/1000:.1f}s do {gyro_packets[-1].ts/1000:.1f}s  ({(gyro_packets[-1].ts - gyro_packets[0].ts)/1000:.1f}s skupaj)"
    )
    print(
        f"Acc:  od {acc_packets[0].ts/1000:.1f}s do {acc_packets[-1].ts/1000:.1f}s  ({(acc_packets[-1].ts - acc_packets[0].ts)/1000:.1f}s skupaj)"
    )
    print(
        f"Mag:  od {mag_packets[0].ts/1000:.1f}s do {mag_packets[-1].ts/1000:.1f}s  ({(mag_packets[-1].ts - mag_packets[0].ts)/1000:.1f}s skupaj)"
    )

    Fvz_gyro, sig_gyro = sestavi_podatke(gyro_packets)
    Fvz_acc, sig_acc = sestavi_podatke(acc_packets)
    Fvz_mag, sig_mag = sestavi_podatke(mag_packets)

    print(f"Giroskop:      {len(sig_gyro)} vzorcev, Fvz = {Fvz_gyro:.2f} Hz")
    print(f"Pospeskometer: {len(sig_acc)} vzorcev,  Fvz = {Fvz_acc:.2f} Hz")
    print(f"Magnetometer:  {len(sig_mag)} vzorcev,  Fvz = {Fvz_mag:.2f} Hz")

    prikazi_signal(
        sig_gyro,
        title=f"Giroskop - celoten signal | Fvz = {Fvz_gyro:.2f} Hz",
        unit=UNITS[ID_GYRO],
        Fvz=Fvz_gyro,
    )
    prikazi_signal(
        sig_acc,
        title=f"Pospeskometer - celoten signal | Fvz = {Fvz_acc:.2f} Hz",
        unit=UNITS[ID_ACC],
        Fvz=Fvz_acc,
    )
    prikazi_signal(
        sig_mag,
        title=f"Magnetometer - celoten signal | Fvz = {Fvz_mag:.2f} Hz",
        unit=UNITS[ID_MAG],
        Fvz=Fvz_mag,
    )

    t_start = 85.0
    t_end = 140.0
    i_start = int(t_start * Fvz_gyro)
    i_end = int(t_end * Fvz_gyro)

    prikazi_signal(
        sig_gyro,
        title=f"Giroskop - interval {t_start}s - {t_end}s | Fvz = {Fvz_gyro:.2f} Hz",
        start_i=i_start,
        end_i=i_end,
        unit=UNITS[ID_GYRO],
        Fvz=Fvz_gyro,
    )
