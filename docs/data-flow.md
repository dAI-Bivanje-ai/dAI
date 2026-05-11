# Pretok podatkov

Projekt uporablja podatke iz IMU senzorja za spremljanje aktivnosti uporabnika.

## Postopek

1. STM32 zajema podatke senzorja.
2. Python skripta `data_logger.py` bere podatke preko serijske komunikacije.
3. Podatki se shranijo za nadaljnjo obdelavo.
4. `label_maker.py` omogoča označevanje aktivnosti.
5. `vizualizacija.py` prikaže rezultate in grafe.

## Cilj

Cilj projekta je analiza delovnih navad in aktivnosti med uporabo računalnika.
