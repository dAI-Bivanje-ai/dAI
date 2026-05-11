# Data Flow

## Opis

Projekt uporablja IMU podatke iz STM32 naprave za zaznavanje aktivnosti uporabnika.
Podatki potujejo skozi več faz obdelave, od zajema signalov do priprave dataseta za strojno učenje.

---

## Pretok podatkov

```mermaid
graph TD

    A[STM32 + IMU Sensor] --> B[data_logger.py]

    B --> C[Raw .bin files]

    C --> D[Packet parsing]

    D --> E[Signal reconstruction]

    E --> F[vizualizacija.py]

    F --> G[label_maker.py]

    G --> H[Labeled intervals]

    H --> I[windower.py]

    I --> J[Signal windows]

    J --> K[stft.py]

    K --> L[Spectrograms]

    L --> M[dataset_builder.py]

    M --> N[Dataset .npz]

    N --> O[ML Model]

    O --> P[Activity classification]