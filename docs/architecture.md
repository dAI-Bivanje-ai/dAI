## Arhitektura sistema

Projekt je razdeljen na več logičnih delov:

### Data Logger

Modul `src/data_logger/` skrbi za zajem in parsanje podatkov iz STM32 naprave. Podatki se berejo preko serijske komunikacije, preverijo in shranijo za nadaljnjo obdelavo.

### Labeling

Modul `src/labeling/` omogoča ročno označevanje aktivnosti uporabnika na podlagi signalov. Trenutne aktivnosti so tipkanje, mirovanje, uporaba telefona, uporaba miške in premik.

### Visualization

Modul `src/visualization/` skrbi za prikaz signalov iz senzorjev. Uporablja se za vizualno preverjanje podatkov in analizo gibanja.

### Preprocessing

Modul `src/preprocessing/` pripravi podatke za strojno učenje. Signal se najprej razreže na časovna okna, nato se iz vsakega okna izračuna spektrogram. Končni rezultat je dataset, ki se lahko uporabi za treniranje modela.

### Model

Modul `src/model/` bo v prihodnje vseboval modele za klasifikacijo aktivnosti uporabnika iz IMU podatkov.

graph TD
    STM32 --> data_logger
    data_logger --> visualization
    visualization --> labeling
    labeling --> preprocessing
    preprocessing --> ML_model
    ML_model --> analytics