# dAI

**dAI** je sistem za spremljanje in prepoznavanje aktivnosti uporabnika pri delu za računalnikom. Prototip združuje podatke iz zapestne naprave, mikrofona in trenutno aktivnega okna ter uporabniku v realnem času prikaže oceno aktivnosti in produktivnosti.

## Opis projekta

Namen projekta je pokazati, da samo merjenje časa pred računalnikom ni dovolj za oceno uporabnikove aktivnosti. Uporabnik ima lahko odprto delovno aplikacijo, vendar v resnici miruje, uporablja telefon ali pa je njegovo delo moteno zaradi zvoka v okolici.

Sistem zato uporablja več virov podatkov:

* **IMU senzorje** za zaznavanje gibanja zapestja,
* **mikrofon** za zaznavanje zvočnega konteksta,
* **aktivno okno** za razvrščanje trenutne aplikacije med produktivno, neproduktivno ali neznano,
* **klasifikacijske modele** za prepoznavanje aktivnosti in zvočnih stanj,
* **GUI** za prikaz rezultatov v realnem času.

## Glavne funkcionalnosti

* zajem podatkov iz naprave STM32,
* razčlenjevanje binarnih podatkovnih paketov,
* obdelava podatkov pospeškomera in giroskopa,
* obdelava mikrofonskih podatkov,
* pretvorba signalov v spektrograme,
* učenje in uporaba klasifikacijskih modelov,
* realnočasovna napoved aktivnosti,
* spremljanje trenutno aktivnega okna,
* izračun produktivnega in neproduktivnega časa,
* prikaz rezultatov v uporabniškem vmesniku,
* osnovni CI/CD pipeline za preverjanje kode.

## Uporabljene tehnologije

* **STM32F411** — zajem senzorskih podatkov,
* **Python 3.11+** — razvoj aplikacije in obdelava podatkov,
* **NumPy** — delo s podatkovnimi polji,
* **SciPy** — obdelava signalov in filtriranje,
* **Matplotlib** — vizualizacija signalov in spektrogramov,
* **PyTorch** — učenje in izvajanje modelov,
* **pyserial** — serijska komunikacija z napravo,
* **pywinctl** — spremljanje aktivnega okna,
* **customtkinter** — grafični uporabniški vmesnik,
* **pytest** — testiranje,
* **GitHub Actions / CI/CD** — avtomatsko preverjanje kode.

## Osnovna struktura projekta

```text
dAI/
├── src/
│   ├── data_logger/        # zajem in shranjevanje podatkov
│   ├── preprocessing/      # obdelava signalov in priprava spektrogramov
│   ├── model/              # modeli, trening in evalvacija
│   ├── realtime/           # realtime zajem, napovedi in GUI
│   └── visualization/      # vizualizacija signalov in rezultatov
├── tests/                  # testi
├── models/                 # shranjeni naučeni modeli
├── docs/                   # dodatna dokumentacija, če je prisotna
├── requirements.txt        # Python odvisnosti
└── README.md
```

## Namestitev

Projekt najprej kloniramo:

```bash
git clone https://github.com/dAI-Bivanje-ai/dAI.git
cd dAI
```

Ustvarimo in aktiviramo virtualno okolje.

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Namestimo odvisnosti:

```bash
pip install -r requirements.txt
```

## Zagon sistema

Za realnočasovno delovanje je treba priključiti napravo STM32. Sistem preko serijske povezave prejema podatke iz IMU senzorjev in mikrofona.

Primer zagona realtime aplikacije CLI:

```bash
python -m src.realtime.main_rt
```

Če se uporablja GUI:

```bash
python -m src.realtime.gui_rt
```

Točen ukaz je odvisen od trenutne strukture entrypointov v repozitoriju.

## Trening modelov

Modeli se učijo iz vnaprej posnetih podatkov. Oznaka aktivnosti je določena že pri snemanju oziroma iz imena datoteke ali strukture map.

Primer zagona treninga IMU modela:

```bash
python -m src.model.train
```

Primer zagona treninga mikrofonskega modela:

```bash
python -m src.model.train_mic
```

## Testiranje

Teste zaženemo z ukazom:

```bash
pytest
```

Projekt vsebuje tudi CI/CD pipeline, ki ob spremembah kode preveri osnovno pravilnost projekta. CI del izvaja teste, CD del pa je pripravljen za build/deploy fazo na release veji.

## Dokumentacija

Podrobnejša zaključna dokumentacija je pripravljena v Wiki obliki. Vključuje:

* projektne specifikacije,
* navodila za namestitev in uporabo,
* ključne primere uporabe,
* izvedene lastnosti,
* znane omejitve in nadaljnje delo,
* zapisnike sestankov.

## Avtorja

* Jan Garmuš
* Jan Kavcl

**Mentor:** izr. prof. dr. Božidar Potočnik
