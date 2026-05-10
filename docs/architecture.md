# Arhitektura projekta

Projekt je razdeljen na več glavnih delov.

## data_logger.py

Skrbi za:
- branje podatkov iz IMU senzorja,
- serijsko komunikacijo,
- shranjevanje podatkov.

## label_maker.py

Skrbi za:
- označevanje aktivnosti,
- pripravo podatkov za nadaljnjo analizo.

## vizualizacija.py

Skrbi za:
- prikaz grafov,
- vizualno analizo podatkov.

## Pretok podatkov

```mermaid
graph TD
    STM32 --> data_logger
    data_logger --> label_maker
    label_maker --> vizualizacija
