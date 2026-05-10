# Pregled projekta

dAI je projekt za spremljanje in analizo delovnih navad pri delu za računalnikom.

Sistem uporablja IMU senzor na STM32F411 za zajem gibanja in aktivnosti uporabnika. Zbrani podatki se nato shranijo, označijo in vizualizirajo v Pythonu.

## Glavni deli projekta

- zajem podatkov iz senzorja,
- shranjevanje podatkov,
- označevanje aktivnosti,
- vizualizacija rezultatov.

## Uporabljene tehnologije

- STM32F411
- Python 3.11+
- NumPy
- Matplotlib
- pyserial
